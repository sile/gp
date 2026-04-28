use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidInput(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
