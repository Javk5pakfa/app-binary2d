use std::time::Instant;
use serde::{Serialize, Deserialize};




/**
 * A task, or side-effect, such as reporting, analysis, or data output
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct RecurringTask {

    /// The number of times this task has been performed
    pub count: usize,

    /// The next simulation time at which this task is set to be performed
    pub next_time: f64,

    /// The last clock time when this task was performed
    #[serde(skip, default = "Instant::now")]
    pub last_performed: Instant,

    #[serde(skip, default = "usize::default")]
    pub count_this_run: usize,
}




/**
 * All the tasks that are used in this application
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct Tasks {

    /// Write a snapshot of the full simulation
    pub write_checkpoint: RecurringTask,

    /// Print the loop message
    pub iteration_message: RecurringTask,

    /// Record a time series sample
    pub record_time_series: RecurringTask,

    /// Summarize the simulation performance
    pub report_progress: RecurringTask,

    /// Only to keep track of the simulation runtime
    pub simulation_startup: RecurringTask,
}




// ============================================================================
impl RecurringTask {

    /**
     * Create a fresh recurring task which is first due at t = 0.0.
     */
    pub fn new() -> Self {
        Self {
            count: 0,
            next_time: 0.0,
            last_performed: Instant::now(),
            count_this_run: 0,
        }
    }

    /**
     * Mark the task as having just been performed, and schedule it to happen
     * again after the given time interval. Return the length of WALL time that
     * elapsed since the task was last performed.
     */
    pub fn advance(&mut self, interval: f64) -> f64 {
        let seconds = self.last_performed.elapsed().as_secs_f64();
        self.count += 1;
        self.count_this_run += 1;
        self.next_time += interval;
        self.last_performed = Instant::now();
        seconds
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.last_performed.elapsed().as_secs_f64()
    }

    pub fn elapsed_hours(&self) -> f64 {
        self.last_performed.elapsed().as_secs_f64() / 3600.0
    }
}




// ============================================================================
impl Tasks {
    pub fn new() -> Self {
        Self {
            write_checkpoint: RecurringTask::new(),
            iteration_message: RecurringTask::new(),
            record_time_series: RecurringTask::new(),
            report_progress: RecurringTask::new(),
            simulation_startup: RecurringTask::new(),
        }
    }
}
