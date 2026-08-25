//! Checkpoint items shared by more than one admitted target.
//!
//! Membership rule: an item lives here when items belonging to two or more targets reach it, or
//! when a target-independent module reaches it. Everything reached by exactly one target lives in
//! that target's directory instead.

pub(crate) mod config_util;
pub(crate) mod inventory;
pub(crate) mod modelopt_codec;
pub(crate) mod mtp;
pub(crate) mod naming;
pub(crate) mod nvfp4;
pub(crate) mod routes;
pub(crate) mod scale_swizzle;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod vision;
