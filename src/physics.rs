use serde::{Serialize, Deserialize};
use godunov_core::runge_kutta;
use kepler_two_body::{
    OrbitalElements,
    OrbitalState,
};
use crate::app::{
    AnyPrimitive,
};
use crate::mesh::{
    Mesh,
};
use crate::state::{
    ItemizedChange,
};
use crate::traits::{
    Arithmetic,
    Conserved,
    Hydrodynamics,
    Primitive,
    Zeros,
};




// ============================================================================
#[derive(thiserror::Error, Debug, Clone)]
pub enum HydroErrorType {
    #[error("negative surface density {0:.4e}")]
    NegativeDensity(f64)
}

impl HydroErrorType {
    pub fn at_position(self, position: (f64, f64)) -> HydroError {
        HydroError{source: self, binary: None, position}
    }
}




// ============================================================================
#[derive(thiserror::Error, Debug, Clone)]
#[error("at position ({:.4} {:.4}), when the binary was at ({:.4} {:.4}) ({:.4} {:.4})",
    position.0,
    position.1,
    binary.map_or(0.0, |s| s.0.position_x()),
    binary.map_or(0.0, |s| s.0.position_y()),
    binary.map_or(0.1, |s| s.1.position_x()),
    binary.map_or(0.1, |s| s.1.position_y()),
)]
pub struct HydroError {
    source: HydroErrorType,
    binary: Option<kepler_two_body::OrbitalState>,
    position: (f64, f64),
}

impl HydroError {
    pub fn with_orbital_state(self, binary: kepler_two_body::OrbitalState) -> Self {
        Self {
            source: self.source,
            binary: Some(binary),
            position: self.position,
        }
    }
}




// ============================================================================
#[derive(Copy, Clone)]
pub struct CellData<'a, P: Primitive> {
    pub pc: &'a P,
    pub gx: &'a P,
    pub gy: &'a P,
}




// ============================================================================
#[derive(Copy, Clone)]
pub enum Direction {
    X,
    Y,
}




// ============================================================================
#[derive(Clone, Serialize, Deserialize)]
pub struct Solver {
    pub buffer_rate: f64,
    pub buffer_scale: f64,
    pub cfl: f64,
    pub domain_radius: f64,
    pub mach_number: f64,
    pub nu: f64,
    pub lambda: f64,
    pub plm: f64,
    pub rk_order: i64,
    pub sink_radius: f64,
    pub sink_rate: f64,
    pub softening_length: f64,
    pub force_flux_comm: bool,
    pub orbital_elements: OrbitalElements,
    pub relative_density_floor: f64,
    pub relative_fake_mass_rate: f64,
}

pub type Physics = Solver;




// ============================================================================
pub struct SourceTerms {
    pub fx1: f64,
    pub fy1: f64,
    pub fx2: f64,
    pub fy2: f64,
    pub sink_rate1: f64,
    pub sink_rate2: f64,
    pub buffer_rate: f64,
}




// ============================================================================
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Isothermal {
    pub mach_number: f64
}




// ============================================================================
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Euler {
    pub gamma_law_index: f64,
}




// ============================================================================
impl<'a, P: Primitive> CellData<'_, P> {

    pub fn new(pc: &'a P, gx: &'a P, gy: &'a P) -> CellData<'a, P> {
        CellData{
            pc: pc,
            gx: gx,
            gy: gy,
        }
    }

    pub fn stress_field(&self, nu: f64, lambda: f64, dx: f64, dy: f64, row: Direction, col: Direction) -> f64 {
        use Direction::{X, Y};

        let shear_stress = match (row, col) {
            (X, X) => 4.0 / 3.0 * self.gx.velocity_x() / dx - 2.0 / 3.0 * self.gy.velocity_y() / dy,
            (X, Y) => 1.0 / 1.0 * self.gx.velocity_y() / dx + 1.0 / 1.0 * self.gy.velocity_x() / dy,
            (Y, X) => 1.0 / 1.0 * self.gx.velocity_y() / dx + 1.0 / 1.0 * self.gy.velocity_x() / dy,
            (Y, Y) =>-2.0 / 3.0 * self.gx.velocity_x() / dx + 4.0 / 3.0 * self.gy.velocity_y() / dy,
        };

        let bulk_stress = match (row, col) {
            (X, X) => self.gx.velocity_x() / dx + self.gy.velocity_y() / dy,
            (X, Y) => 0.0,
            (Y, X) => 0.0,
            (Y, Y) => self.gx.velocity_x() / dx + self.gy.velocity_y() / dy,
        };

        self.pc.mass_density() * (nu * shear_stress + lambda * bulk_stress)
    }

    pub fn gradient_field(&self, axis: Direction) -> &P {
        use Direction::{X, Y};
        match axis
        {
            X => self.gx,
            Y => self.gy,
        }
    }
}




// ============================================================================
impl Solver {

    pub fn runge_kutta(&self) -> runge_kutta::RungeKuttaOrder {
        use std::convert::TryFrom;
        runge_kutta::RungeKuttaOrder::try_from(self.rk_order).expect("illegal RK order")
    }

    pub fn need_flux_communication(&self) -> bool {
        self.force_flux_comm
    }

    pub fn effective_resolution(&self, mesh: &Mesh) -> f64 {
        f64::min(mesh.cell_spacing_x(), mesh.cell_spacing_y())
    }

    pub fn min_time_step(&self, mesh: &Mesh) -> f64 {
        self.cfl * self.effective_resolution(mesh) / self.maximum_orbital_velocity()
    }

    pub fn sink_kernel(&self, dx: f64, dy: f64) -> f64 {
        let r2 = dx * dx + dy * dy;
        let s2 = self.sink_radius * self.sink_radius;

        if r2 < s2 * 9.0 {
            self.sink_rate * f64::exp(-(r2 / s2).powi(3))
        } else {
            0.0
        }
    }

    pub fn sound_speed_squared(&self, xy: &(f64, f64), state: &OrbitalState) -> f64 {
        -state.gravitational_potential(xy.0, xy.1, self.softening_length) / self.mach_number.powi(2)
    }

    pub fn maximum_orbital_velocity(&self) -> f64 {
        1.0 / self.softening_length.sqrt()
    }

    pub fn source_terms(&self, two_body_state: &kepler_two_body::OrbitalState, x: f64, y: f64, surface_density: f64) -> SourceTerms {
        let p1 = two_body_state.0;
        let p2 = two_body_state.1;

        let [ax1, ay1] = p1.gravitational_acceleration(x, y, self.softening_length);
        let [ax2, ay2] = p2.gravitational_acceleration(x, y, self.softening_length);

        let fx1 = surface_density * ax1;
        let fy1 = surface_density * ay1;
        let fx2 = surface_density * ax2;
        let fy2 = surface_density * ay2;

        let x1 = p1.position_x();
        let y1 = p1.position_y();
        let x2 = p2.position_x();
        let y2 = p2.position_y();

        let sink_rate1 = self.sink_kernel(x - x1, y - y1);
        let sink_rate2 = self.sink_kernel(x - x2, y - y2);

        let r = (x * x + y * y).sqrt();
        let y = (r - self.domain_radius) / self.buffer_scale;
        let omega_outer = (two_body_state.total_mass() / self.domain_radius.powi(3)).sqrt();
        let buffer_rate = 0.5 * self.buffer_rate * (1.0 + f64::tanh(y)) * omega_outer;

        SourceTerms {
            fx1: fx1,
            fy1: fy1,
            fx2: fx2,
            fy2: fy2,
            sink_rate1: sink_rate1,
            sink_rate2: sink_rate2,
            buffer_rate: buffer_rate,
        }
    }

    pub fn relative_density_floor(&self) -> f64 {
        self.relative_density_floor
    }

    pub fn relative_fake_mass_rate(&self) -> f64 {
        self.relative_fake_mass_rate
    }
}




// ============================================================================
impl Isothermal {
    pub fn new(mach_number: f64) -> Self {
        Self{mach_number}
    }
}




// ============================================================================
impl Hydrodynamics for Isothermal
{
    type Conserved = hydro_iso2d::Conserved;
    type Primitive = hydro_iso2d::Primitive;

    fn gamma_law_index(&self) -> f64 {
        1.0
    }

    fn global_mach_number(&self) -> Option<f64> {
        Some(self.mach_number)
    }

    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive {
        godunov_core::piecewise_linear::plm_gradient3(theta, a, b, c)
    }

    fn try_to_primitive(&self, u: Self::Conserved) -> Result<Self::Primitive, HydroErrorType> {
        if u.density() < 0.0 {
            return Err(HydroErrorType::NegativeDensity(u.density()))
        }
        Ok(u.to_primitive())
    }

    fn to_primitive(&self, u: Self::Conserved) -> Self::Primitive {
        u.to_primitive()
    }

    fn to_conserved(&self, p: Self::Primitive) -> Self::Conserved {
        p.to_conserved()
    }

    fn from_any(&self, p: AnyPrimitive) -> Self::Primitive {
        hydro_iso2d::Primitive(
            p.surface_density,
            p.velocity_x,
            p.velocity_y,
        )
    }

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &kepler_two_body::OrbitalState) -> ItemizedChange<Self::Conserved>
    {
        let omega = 1.0; // Note: in the future, the binary orbital frequency
                         // may be allowed to vary; we really should not be
                         // assuming everywhere that it's 1.0.
        let density_floor = background_conserved.density() * solver.relative_density_floor();
        let fake_mass_rate = background_conserved.density() * omega * solver.relative_fake_mass_rate();

        let fake_mdot = if conserved.density() < density_floor {
            0.0
        } else {
            fake_mass_rate
        };

        let st = solver.source_terms(two_body_state, x, y, conserved.density());

        ItemizedChange {
            grav1:   hydro_iso2d::Conserved(0.0, st.fx1, st.fy1) * dt,
            grav2:   hydro_iso2d::Conserved(0.0, st.fx2, st.fy2) * dt,
            sink1:   conserved * (-st.sink_rate1 * dt),
            sink2:   conserved * (-st.sink_rate2 * dt),
            buffer: (conserved - background_conserved) * (-dt * st.buffer_rate),
            cooling: Self::Conserved::zeros(),
            fake_mass: hydro_iso2d::Conserved(fake_mdot * dt, 0.0, 0.0),
        }
    }

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, hydro_iso2d::Primitive>,
        r: &CellData<'a, hydro_iso2d::Primitive>,
        f: &(f64, f64),
        dx: f64,
        dy: f64,
        two_body_state: &kepler_two_body::OrbitalState,
        axis: Direction) -> hydro_iso2d::Conserved
    {
        let cs2 = solver.sound_speed_squared(f, &two_body_state);
        let pl  = *l.pc + *l.gradient_field(axis) * 0.5;
        let pr  = *r.pc - *r.gradient_field(axis) * 0.5;
        let nu  = solver.nu;
        let lam = solver.lambda;
        let tau_x = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::X) + r.stress_field(nu, lam, dx, dy, axis, Direction::X));
        let tau_y = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::Y) + r.stress_field(nu, lam, dx, dy, axis, Direction::Y));
        let iso2d_axis = match axis {
            Direction::X => hydro_iso2d::Direction::X,
            Direction::Y => hydro_iso2d::Direction::Y,
        };
        hydro_iso2d::riemann_hlle(pl, pr, iso2d_axis, cs2) + hydro_iso2d::Conserved(0.0, -tau_x, -tau_y)
    }
}




// ============================================================================
impl Euler {
    pub fn new() -> Self {
        Self{gamma_law_index: 5.0 / 3.0}
    }
}




// ============================================================================
impl Hydrodynamics for Euler {

    type Conserved = hydro_euler::euler_2d::Conserved;
    type Primitive = hydro_euler::euler_2d::Primitive;

    fn gamma_law_index(&self) -> f64 {
        self.gamma_law_index
    }

    fn global_mach_number(&self) -> Option<f64> {
        None
    }

    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive {
        godunov_core::piecewise_linear::plm_gradient4(theta, a, b, c)
    }

    fn try_to_primitive(&self, u: Self::Conserved) -> Result<Self::Primitive, HydroErrorType> {
        if u.mass_density() < 0.0 {
            return Err(HydroErrorType::NegativeDensity(u.mass_density()))
        }
        Ok(u.to_primitive(self.gamma_law_index))
    }

    fn to_primitive(&self, conserved: Self::Conserved) -> Self::Primitive {
        conserved.to_primitive(self.gamma_law_index)
    }

    fn to_conserved(&self, p: Self::Primitive) -> Self::Conserved {
        p.to_conserved(self.gamma_law_index)
    }

    fn from_any(&self, p: AnyPrimitive) -> Self::Primitive {
        hydro_euler::euler_2d::Primitive(
            p.surface_density,
            p.velocity_x,
            p.velocity_y,
            p.surface_pressure,
        )
    }

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &kepler_two_body::OrbitalState) -> ItemizedChange<Self::Conserved>
    {
        let st        = solver.source_terms(two_body_state, x, y, conserved.mass_density());
        let primitive = conserved.to_primitive(self.gamma_law_index);
        let vx        = primitive.velocity_1();
        let vy        = primitive.velocity_2();

        ItemizedChange {
            grav1:   hydro_euler::euler_2d::Conserved(0.0, st.fx1, st.fy1, st.fx1 * vx + st.fy1 * vy) * dt,
            grav2:   hydro_euler::euler_2d::Conserved(0.0, st.fx2, st.fy2, st.fx2 * vx + st.fy2 * vy) * dt,
            sink1:   conserved * (-st.sink_rate1 * dt),
            sink2:   conserved * (-st.sink_rate2 * dt),
            buffer: (conserved - background_conserved) * (-dt * st.buffer_rate),
            cooling: Self::Conserved::zeros(),
            fake_mass: Self::Conserved::zeros(),
        }
    }

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>,
        r: &CellData<'a, Self::Primitive>,
        _: &(f64, f64),
        dx: f64,
        dy: f64,
        _: &kepler_two_body::OrbitalState,
        axis: Direction) -> Self::Conserved
    {
        let pl = *l.pc + *l.gradient_field(axis) * 0.5;
        let pr = *r.pc - *r.gradient_field(axis) * 0.5;

        let nu    = solver.nu;
        let lam   = solver.lambda;
        let tau_x = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::X) + r.stress_field(nu, lam, dx, dy, axis, Direction::X));
        let tau_y = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::Y) + r.stress_field(nu, lam, dx, dy, axis, Direction::Y));
        let vx = 0.5 * (l.pc.velocity_x() + r.pc.velocity_x());
        let vy = 0.5 * (l.pc.velocity_y() + r.pc.velocity_y());
        let viscous_flux = hydro_euler::euler_2d::Conserved(0.0, -tau_x, -tau_y, -(tau_x * vx + tau_y * vy));

        let euler_axis = match axis {
            Direction::X => hydro_euler::geometry::Direction::X,
            Direction::Y => hydro_euler::geometry::Direction::Y,
        };
        hydro_euler::euler_2d::riemann_hlle(pl, pr, euler_axis, self.gamma_law_index) + viscous_flux
    }
}




// ============================================================================
impl Arithmetic for hydro_iso2d::Conserved {}
impl Arithmetic for hydro_euler::euler_2d::Conserved {}
// impl Arithmetic for kepler_two_body::OrbitalElements {}




// ============================================================================
impl Zeros for hydro_iso2d::Conserved {
    fn zeros() -> Self {
        Self(0.0, 0.0, 0.0)
    }
}

impl Zeros for hydro_euler::euler_2d::Conserved {
    fn zeros() -> Self {
        Self(0.0, 0.0, 0.0, 0.0)
    }
}

impl Zeros for kepler_two_body::OrbitalElements {
    fn zeros() -> Self {
        Self(0.0, 0.0, 0.0, 0.0)
    }
}




// ============================================================================
impl Conserved for hydro_iso2d::Conserved {
    fn mass_and_momentum(&self) -> (f64, f64, f64) {
        (self.0, self.1, self.2)
    }
}

impl Conserved for hydro_euler::euler_2d::Conserved {
    fn mass_and_momentum(&self) -> (f64, f64, f64) {
        (self.0, self.1, self.2)
    }
}




// ============================================================================
impl Primitive for hydro_iso2d::Primitive {
    fn velocity_x(self)   -> f64 { self.velocity_x() }
    fn velocity_y(self)   -> f64 { self.velocity_y() }
    fn mass_density(self) -> f64 { self.density() }
}

impl Primitive for hydro_euler::euler_2d::Primitive {
    fn velocity_x(self)   -> f64 { self.velocity(hydro_euler::geometry::Direction::X) }
    fn velocity_y(self)   -> f64 { self.velocity(hydro_euler::geometry::Direction::Y) }
    fn mass_density(self) -> f64 { self.mass_density() }
}
