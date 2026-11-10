// Convert diffsol errors to custom pydiffsol error type

use diffsol::error::DiffsolError;
use std::fmt;

#[derive(Debug)]
pub struct DiffsolRtError(DiffsolError);

impl fmt::Display for DiffsolRtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DiffsolRtError {}

impl DiffsolRtError {
    /// The same error, with `name` (for example the name of an argument) at the start of the
    /// message.
    pub(crate) fn named(self, name: &str) -> Self {
        let message = match self.0 {
            DiffsolError::Other(message) => message,
            other => other.to_string(),
        };
        DiffsolError::Other(format!("{name}: {message}")).into()
    }
}

impl From<DiffsolError> for DiffsolRtError {
    fn from(other: DiffsolError) -> Self {
        DiffsolRtError(other)
    }
}
