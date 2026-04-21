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

impl From<DiffsolError> for DiffsolRtError {
    fn from(other: DiffsolError) -> Self {
        DiffsolRtError(other)
    }
}
