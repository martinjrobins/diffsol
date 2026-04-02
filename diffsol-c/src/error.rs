// Convert diffsol errors to custom pydiffsol error type

use diffsol::error::DiffsolError;
use std::fmt;

#[derive(Debug)]
pub struct DiffsolJsError(DiffsolError);

impl fmt::Display for DiffsolJsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DiffsolJsError {}

impl From<DiffsolError> for DiffsolJsError {
    fn from(other: DiffsolError) -> Self {
        DiffsolJsError(other)
    }
}
