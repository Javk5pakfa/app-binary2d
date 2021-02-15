use serde::{Serialize, Deserialize};
use crate::scheme::State;
use crate::traits::Hydrodynamics;
use crate::physics::{Isothermal, Euler};
use crate::mesh::Mesh;




/**
 * Either of the supported hydro systems 
 */
#[derive(Clone, Serialize, Deserialize, derive_more::From)]
pub enum AnyHydro {
    Isothermal (Isothermal),
    Euler      (Euler),
}




/**
 * Either of the simulation state types corresponding to the different hydro
 * systems 
 */
#[derive(Clone, Serialize, Deserialize, derive_more::From)]
pub enum AnyState {
    Isothermal (State<<Isothermal as Hydrodynamics>::Conserved>),
    Euler      (State<<Euler      as Hydrodynamics>::Conserved>),
}




/**
 * Description of the hydrodynamics state that is compatible with any of the
 * supported hydro systems
 */
#[derive(Serialize, Deserialize)]
struct AnyPrimitive {

    /// X-component of gas velocity
    velocity_x: f64,

    /// Y-component of gas velocity
    velocity_y: f64,

    /// Vertically integrated mass density, Sigma
    surface_density: f64,

    /// Vertically integrated gas pressure, P
    surface_pressure: f64,
}




/**
 * Simulation control: how long to run for, how frequently to perform side
 * effects, etc
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct Control {

    /// Number of orbits the simulation will be run to
    pub num_orbits: f64,

    /// Number of orbits between writing checkpoints
    pub checkpoint_interval: f64,

    /// Number of iterations between performing side effects. Larger values
    /// yield less terminal output and more accurate performance estimates.
    pub fold: usize,

    /// Number of worker threads on the Tokio runtime
    pub num_threads: usize,
}




/**
 * Runtime configuration
 */
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub hydro: AnyHydro,
    // pub model: Model,
    pub mesh: Mesh,
    pub control: Control,
}




/**
 * App state
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct App {
    pub state: AnyState,
    // pub tasks: Tasks,
    pub config: Configuration,
    pub version: String,
}
