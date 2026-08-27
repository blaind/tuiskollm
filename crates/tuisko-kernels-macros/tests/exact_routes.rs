//! Compile-time and behavioral coverage for the generated exact-route table.

use tuisko_kernels_macros::ExactRoutes;

struct Module;

struct PreparedRoute<const ROWS: usize>;

impl<const ROWS: usize> PreparedRoute<ROWS> {
    fn prepare(_module: &Module) -> Result<Self, &'static str> {
        Ok(Self)
    }

    fn ptx_names() -> Vec<&'static str> {
        vec![match ROWS {
            1 => "one",
            32 => "thirty_two",
            _ => "unexpected",
        }]
    }

    fn value(&self) -> usize {
        ROWS
    }
}

#[derive(ExactRoutes)]
#[exact_routes(
    module(Module),
    error(&'static str),
    dispatch(dispatch_test_route),
    required(1)
)]
struct Routes<const LARGE: bool> {
    #[route(1)]
    one: PreparedRoute<1>,
    #[route(32, admitted(LARGE))]
    thirty_two: PreparedRoute<32>,
}

#[test]
fn prepares_and_dispatches_concrete_routes() {
    let routes = Routes::<true>::prepare(&Module).unwrap();

    assert_eq!(
        dispatch_test_route!(&routes, 1, |route| route.value(), else => 0),
        1
    );
    assert_eq!(
        dispatch_test_route!(&routes, 32, |route| route.value(), else => 0),
        32
    );
    assert_eq!(
        dispatch_test_route!(&routes, 2, |route| route.value(), else => 0),
        0
    );
}

#[test]
fn conditional_route_controls_admission_dispatch_and_inventory() {
    let routes = Routes::<false>::prepare(&Module).unwrap();

    assert_eq!(Routes::<false>::admitted_rows(), vec![1]);
    assert_eq!(Routes::<false>::ptx_names(), vec!["one"]);
    assert!(!Routes::<false>::contains(32));
    assert_eq!(
        dispatch_test_route!(&routes, 32, |route| route.value(), else => 0),
        0
    );

    assert_eq!(Routes::<true>::admitted_rows(), vec![1, 32]);
    assert_eq!(Routes::<true>::ptx_names(), vec!["one", "thirty_two"]);
}
