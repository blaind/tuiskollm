//! Derives explicit, statically dispatched exact-route tables for Tuisko kernels.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use std::collections::BTreeMap;
use syn::{
    Data, DeriveInput, Expr, Fields, Ident, LitBool, LitInt, Token, Type,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
};

/// Derives preparation, admission, optional inventory aggregation, and static dispatch.
///
/// The input must be a named-field struct with one `#[route(ROWS)]` attribute
/// on every field. An optional admission expression may be supplied as
/// `#[route(ROWS, admitted(EXPR))]`. The companion `#[exact_routes(...)]`
/// attribute names the module and error types, dispatch macro, required
/// unconditional rows, and whether to aggregate route inventories. Route
/// types provide `prepare` and, when inventory is enabled, `ptx_names`.
#[proc_macro_derive(ExactRoutes, attributes(exact_routes, route))]
pub fn derive_exact_routes(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_exact_routes(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

struct ExactRoutesConfig {
    module: Type,
    error: Type,
    dispatch: Ident,
    required: Vec<usize>,
    inventory: bool,
}

impl Parse for ExactRoutesConfig {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut module = None;
        let mut error = None;
        let mut dispatch = None;
        let mut required = None;
        let mut inventory = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let content;
            syn::parenthesized!(content in input);

            match key.to_string().as_str() {
                "module" => set_once(&mut module, content.parse()?, &key)?,
                "error" => set_once(&mut error, content.parse()?, &key)?,
                "dispatch" => set_once(&mut dispatch, content.parse()?, &key)?,
                "required" => {
                    let values = Punctuated::<LitInt, Token![,]>::parse_terminated(&content)?;
                    let values = values
                        .into_iter()
                        .map(|value| value.base10_parse::<usize>())
                        .collect::<syn::Result<Vec<_>>>()?;
                    set_once(&mut required, values, &key)?;
                }
                "inventory" => {
                    let value = content.parse::<LitBool>()?.value;
                    set_once(&mut inventory, value, &key)?;
                }
                _ => return Err(syn::Error::new(key.span(), "unknown exact_routes option")),
            }

            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(Self {
            module: required_option(module, "module", input.span())?,
            error: required_option(error, "error", input.span())?,
            dispatch: required_option(dispatch, "dispatch", input.span())?,
            required: required_option(required, "required", input.span())?,
            inventory: inventory.unwrap_or(true),
        })
    }
}

fn set_once<T>(slot: &mut Option<T>, value: T, key: &Ident) -> syn::Result<()> {
    if slot.replace(value).is_some() {
        return Err(syn::Error::new(
            key.span(),
            format!("duplicate exact_routes option `{key}`"),
        ));
    }
    Ok(())
}

fn required_option<T>(value: Option<T>, name: &str, span: proc_macro2::Span) -> syn::Result<T> {
    value.ok_or_else(|| syn::Error::new(span, format!("missing exact_routes option `{name}`")))
}

struct RouteAttr {
    rows: usize,
    admitted: Option<Expr>,
}

impl Parse for RouteAttr {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let rows = input.parse::<LitInt>()?.base10_parse()?;
        let admitted = if input.is_empty() {
            None
        } else {
            input.parse::<Token![,]>()?;
            let key: Ident = input.parse()?;
            if key != "admitted" {
                return Err(syn::Error::new(key.span(), "expected `admitted(EXPR)`"));
            }
            let content;
            syn::parenthesized!(content in input);
            let expression = content.parse()?;
            if !input.is_empty() {
                return Err(input.error("unexpected tokens after admission expression"));
            }
            Some(expression)
        };

        Ok(Self { rows, admitted })
    }
}

struct RouteField {
    ident: Ident,
    ty: Type,
    rows: usize,
    admitted: Option<Expr>,
}

fn expand_exact_routes(input: DeriveInput) -> syn::Result<TokenStream2> {
    let config = parse_config(&input)?;
    let fields = parse_fields(&input)?;
    validate_routes(&fields, &config.required, input.ident.span())?;

    let name = &input.ident;
    let ExactRoutesConfig {
        module,
        error,
        dispatch,
        inventory,
        ..
    } = config;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();
    let prepare_fields = fields.iter().map(|field| {
        let ident = &field.ident;
        let ty = &field.ty;
        quote! { #ident: <#ty>::prepare(module)? }
    });
    let contains_arms = fields.iter().map(|field| {
        let rows = field.rows;
        match &field.admitted {
            Some(admitted) => quote! { #rows if #admitted => true },
            None => quote! { #rows => true },
        }
    });
    let admitted_pushes = fields.iter().map(|field| {
        let rows = field.rows;
        match &field.admitted {
            Some(admitted) => quote! {
                if #admitted {
                    rows.push(#rows);
                }
            },
            None => quote! { rows.push(#rows); },
        }
    });
    let dispatch_arms = fields.iter().map(|field| {
        let rows = field.rows;
        let ident = &field.ident;
        quote! {
            #rows => {
                let $route = &__tuisko_exact_routes.#ident;
                $body
            }
        }
    });
    let inventory_method = inventory.then(|| {
        let inventory_extensions = fields.iter().map(|field| {
            let ty = &field.ty;
            match &field.admitted {
                Some(admitted) => quote! {
                    if #admitted {
                        names.extend(<#ty>::ptx_names());
                    }
                },
                None => quote! { names.extend(<#ty>::ptx_names()); },
            }
        });
        quote! {
            fn ptx_names() -> ::std::vec::Vec<&'static str> {
                let mut names = ::std::vec::Vec::new();
                #(#inventory_extensions)*
                names
            }
        }
    });
    let exact_route_count = fields.len();
    let helper = Ident::new("__tuisko_exact_routes_contains", name.span());

    Ok(quote! {
        impl #impl_generics #name #type_generics #where_clause {
            fn prepare(module: &#module) -> ::core::result::Result<Self, #error> {
                Ok(Self {
                    #(#prepare_fields,)*
                })
            }

            fn contains(rows: usize) -> bool {
                match rows {
                    #(#contains_arms,)*
                    _ => false,
                }
            }

            fn #helper(&self, rows: usize) -> bool {
                Self::contains(rows)
            }

            #[cfg(test)]
            fn admitted_rows() -> ::std::vec::Vec<usize> {
                let mut rows = ::std::vec::Vec::with_capacity(#exact_route_count);
                #(#admitted_pushes)*
                rows
            }

            #inventory_method
        }

        macro_rules! #dispatch {
            ($routes:expr, $rows:expr, |$route:ident| $body:expr, else => $fallback:expr $(,)?) => {{
                let __tuisko_exact_routes = $routes;
                let __tuisko_exact_rows = $rows;
                if !__tuisko_exact_routes.#helper(__tuisko_exact_rows) {
                    $fallback
                } else {
                    match __tuisko_exact_rows {
                        #(#dispatch_arms,)*
                        _ => ::core::unreachable!(),
                    }
                }
            }};
        }
    })
}

fn parse_config(input: &DeriveInput) -> syn::Result<ExactRoutesConfig> {
    let mut configs = input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("exact_routes"));
    let config = configs
        .next()
        .ok_or_else(|| syn::Error::new(input.ident.span(), "missing `#[exact_routes(...)]`"))?;
    if let Some(extra) = configs.next() {
        return Err(syn::Error::new(
            extra.span(),
            "expected exactly one `#[exact_routes(...)]` attribute",
        ));
    }
    config.parse_args()
}

fn parse_fields(input: &DeriveInput) -> syn::Result<Vec<RouteField>> {
    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new(
                    data.fields.span(),
                    "ExactRoutes requires a named-field struct",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new(
                input.ident.span(),
                "ExactRoutes can only be derived for structs",
            ));
        }
    };

    fields
        .iter()
        .map(|field| {
            let mut routes = field
                .attrs
                .iter()
                .filter(|attr| attr.path().is_ident("route"));
            let route = routes.next().ok_or_else(|| {
                syn::Error::new(
                    field.span(),
                    "every exact-route field needs `#[route(ROWS)]`",
                )
            })?;
            if let Some(extra) = routes.next() {
                return Err(syn::Error::new(
                    extra.span(),
                    "expected exactly one `#[route(...)]` attribute per field",
                ));
            }
            let route = route.parse_args::<RouteAttr>()?;
            Ok(RouteField {
                ident: field.ident.clone().expect("named fields have identifiers"),
                ty: field.ty.clone(),
                rows: route.rows,
                admitted: route.admitted,
            })
        })
        .collect()
}

fn validate_routes(
    fields: &[RouteField],
    required: &[usize],
    span: proc_macro2::Span,
) -> syn::Result<()> {
    let mut by_rows = BTreeMap::new();
    for field in fields {
        if field.rows == 0 {
            return Err(syn::Error::new(
                field.ident.span(),
                "exact route row count must be positive",
            ));
        }
        if let Some(previous) = by_rows.insert(field.rows, &field.ident) {
            return Err(syn::Error::new(
                field.ident.span(),
                format!(
                    "duplicate exact route {} on fields `{previous}` and `{}`",
                    field.rows, field.ident
                ),
            ));
        }
    }

    let mut required_rows = BTreeMap::new();
    for rows in required {
        if required_rows.insert(*rows, ()).is_some() {
            return Err(syn::Error::new(
                span,
                format!("required exact route {rows} is listed more than once"),
            ));
        }
        let Some(field) = fields.iter().find(|field| field.rows == *rows) else {
            return Err(syn::Error::new(
                span,
                format!("required exact route {rows} is missing"),
            ));
        };
        if field.admitted.is_some() {
            return Err(syn::Error::new(
                field.ident.span(),
                format!("required exact route {rows} cannot be conditional"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::expand_exact_routes;
    use syn::parse_quote;

    #[test]
    fn rejects_duplicate_rows() {
        let error = expand_exact_routes(parse_quote! {
            #[derive(ExactRoutes)]
            #[exact_routes(
                module(Module), error(Error),
                dispatch(dispatch), required(1)
            )]
            struct Routes {
                #[route(1)] first: First,
                #[route(1)] duplicate: Second,
            }
        })
        .unwrap_err();

        assert!(error.to_string().contains("duplicate exact route 1"));
    }

    #[test]
    fn rejects_a_missing_required_route() {
        let error = expand_exact_routes(parse_quote! {
            #[derive(ExactRoutes)]
            #[exact_routes(
                module(Module), error(Error),
                dispatch(dispatch), required(1, 2)
            )]
            struct Routes {
                #[route(1)] first: First,
            }
        })
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("required exact route 2 is missing")
        );
    }

    #[test]
    fn emits_a_static_match_for_every_field() {
        let output = expand_exact_routes(parse_quote! {
            #[derive(ExactRoutes)]
            #[exact_routes(
                module(Module), error(Error),
                dispatch(dispatch), required(1), inventory(false)
            )]
            struct Routes<const LARGE: bool> {
                #[route(1)] first: First,
                #[route(32, admitted(LARGE))] large: Large,
            }
        })
        .unwrap();
        let output = output.to_string();

        assert!(output.contains("1usize =>"));
        assert!(output.contains("32usize =>"));
        assert!(!output.contains("ptx_names"));
    }
}
