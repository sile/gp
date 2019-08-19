use trackable::error::{ErrorKind as TrackableErrorKind, TrackableError};

#[derive(Debug, Clone, TrackableError)]
pub struct Error(TrackableError<ErrorKind>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    InvalidInput,
}
impl TrackableErrorKind for ErrorKind {}
