//! Proc macros for aiform.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, ItemFn, LitStr};

/// Generates JSON schema for tool arguments.
///
/// Supports structs and enums. Fields can use `#[desc("...")]` to add descriptions.
///
/// # Example
///
/// ```ignore
/// #[derive(ToolArg, Deserialize)]
/// struct Args {
///     #[desc("The location to check")]
///     location: String,
///     count: i32,
/// }
/// ```
#[proc_macro_derive(ToolArg)]
pub fn tool_arg_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_tool_arg(&input).into()
}

fn impl_tool_arg(ast: &DeriveInput) -> proc_macro2::TokenStream {
    let name = &ast.ident;
    match &ast.data {
        syn::Data::Struct(s) => impl_tool_arg_struct(name, &s.fields),
        syn::Data::Enum(e) => impl_tool_arg_enum(name, &e.variants, &ast.attrs),
        _ => panic!("ToolArg supports structs and enums"),
    }
}

fn impl_tool_arg_struct(name: &syn::Ident, fields: &syn::Fields) -> proc_macro2::TokenStream {
    let mut properties = vec![];
    let mut required = vec![];

    for field in fields.iter() {
        let ident = field.ident.as_ref().unwrap();
        let ty = &field.ty;
        let desc = get_desc(&field.attrs);
        let field_schema = schema_expr(ty, &desc);
        let ident_str = ident.to_string();

        properties.push(quote! {
            #ident_str: #field_schema
        });

        if !is_option(ty) {
            required.push(quote!(#ident_str));
        }
    }

    let properties_tokens = quote! { #(#properties),* };
    let required_tokens = quote! { #(#required),* };

    quote! {
        impl ToolArg for #name {
            fn schema() -> serde_json::Value {
                serde_json::json!({
                    "type": "object",
                    "properties": { #properties_tokens },
                    "required": [#required_tokens]
                })
            }
        }
    }
}

fn impl_tool_arg_enum(
    name: &syn::Ident,
    variants: &syn::punctuated::Punctuated<syn::Variant, syn::Token![,]>,
    attrs: &[syn::Attribute],
) -> proc_macro2::TokenStream {
    let desc = get_desc(attrs);
    let mut one_of = vec![];

    for variant in variants.iter() {
        let variant_name = variant.ident.to_string();
        let mut properties = vec![quote!("type": serde_json::json!({"const": #variant_name}))];
        let mut required = vec![quote!("type")];

        match &variant.fields {
            syn::Fields::Unit => {
                // Unit variant: just the type
            }
            syn::Fields::Unnamed(fields) => {
                if fields.unnamed.len() == 1 {
                    // Single tuple field
                    let field = &fields.unnamed[0];
                    let field_ty = &field.ty;
                    let value_schema = schema_expr(field_ty, "");
                    properties.push(quote!("value": #value_schema));
                    required.push(quote!("value"));
                } else {
                    // Multiple tuple fields: treat as array
                    let items: Vec<_> = fields
                        .unnamed
                        .iter()
                        .map(|f| schema_expr(&f.ty, ""))
                        .collect();
                    let items_tokens = quote! { #(#items),* };
                    properties.push(quote!("value": serde_json::json!({"type": "array", "items": [#items_tokens]})));
                    required.push(quote!("value"));
                }
            }
            syn::Fields::Named(fields) => {
                // Named fields
                for field in &fields.named {
                    let field_name = field.ident.as_ref().unwrap().to_string();
                    let field_ty = &field.ty;
                    let field_schema = schema_expr(field_ty, "");
                    properties.push(quote!(#field_name: #field_schema));
                    required.push(quote!(#field_name));
                }
            }
        }

        let props_tokens = quote! { #(#properties),* };
        let req_tokens = quote! { #(#required),* };

        one_of.push(quote! {
            serde_json::json!({"type": "object", "properties": {#props_tokens}, "required": [#req_tokens]})
        });
    }

    let one_of_tokens = quote! { #(#one_of),* };
    let desc_expr = if desc.is_empty() {
        quote!()
    } else {
        quote!(, "description": #desc)
    };

    quote! {
        impl ToolArg for #name {
            fn schema() -> serde_json::Value {
                serde_json::json!({"oneOf": [#one_of_tokens] #desc_expr})
            }
        }
    }
}

/// Turns an async function into an OpenAI tool.
///
/// The function must take one parameter that implements `ToolArg`.
///
/// # Example
///
/// ```ignore
/// #[tool("Get the weather for a location")]
/// async fn get_weather(args: WeatherArgs) -> Result<String, Box<dyn Error + Send + Sync>> {
///     // implementation
/// }
/// ```
///
/// This generates a `GetWeatherTool` struct that implements the `Tool` trait.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let desc = if attr.is_empty() {
        String::new()
    } else {
        let lit: LitStr = syn::parse(attr).unwrap();
        lit.value()
    };
    let func = parse_macro_input!(item as ItemFn);
    impl_tool(&func, &desc).into()
}

fn impl_tool(func: &ItemFn, desc: &str) -> proc_macro2::TokenStream {
    let name = &func.sig.ident;
    let param = func
        .sig
        .inputs
        .first()
        .expect("Tool function must have at least one parameter");
    let param_ty = if let syn::FnArg::Typed(p) = param {
        &*p.ty
    } else {
        panic!("Tool function parameters must be typed");
    };

    let pascal_name = to_pascal_case(&name.to_string());
    let tool_struct = syn::Ident::new(&format!("{}Tool", pascal_name), name.span());

    quote! {
        #func

        pub struct #tool_struct;

        impl Tool for #tool_struct {
            const NAME: &'static str = stringify!(#name);
            const DESCRIPTION: &'static str = #desc;

            fn name() -> &'static str {
                Self::NAME
            }

            fn description() -> &'static str {
                Self::DESCRIPTION
            }

            fn parameters() -> serde_json::Value {
                <#param_ty as ToolArg>::schema()
            }

            async fn call(&self, args: serde_json::Value) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                let parsed_args: #param_ty = serde_json::from_value(args)?;
                #name(parsed_args).await
            }
        }
    }
}

/// Generates JSON schema for structured output.
///
/// Use this with OpenAI's structured output feature to get typed responses.
///
/// # Example
///
/// ```ignore
/// #[derive(StructuredOutput, ToolArg, Deserialize)]
/// struct Response {
///     summary: String,
///     confidence: f64,
/// }
/// ```
#[proc_macro_derive(StructuredOutput)]
pub fn structured_output_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_structured_output(&input).into()
}

fn impl_structured_output(ast: &DeriveInput) -> proc_macro2::TokenStream {
    let name = &ast.ident;
    quote! {
        impl StructuredOutput for #name {
            fn schema() -> serde_json::Value {
                // For structured output, we can use the same schema as ToolArg
                // But in practice, OpenAI structured output might require specific format
                // For now, assume similar to ToolArg
                <#name as ToolArg>::schema()
            }
        }
    }
}

// Helper functions

/// Extracts description from #[desc("...")] attribute.
fn get_desc(attrs: &[syn::Attribute]) -> String {
    for attr in attrs {
        if attr.path().is_ident("desc") {
            if let Ok(lit) = attr.parse_args::<syn::Lit>() {
                if let syn::Lit::Str(s) = lit {
                    return s.value();
                }
            }
        }
    }
    String::new()
}

/// Generates a JSON schema expression for a Rust type.
fn schema_expr(ty: &syn::Type, desc: &str) -> proc_macro2::TokenStream {
    let desc_expr = if desc.is_empty() {
        quote!()
    } else {
        quote!(, "description": #desc)
    };

    if let syn::Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            match seg.ident.to_string().as_str() {
                "String" => quote!(serde_json::json!({"type": "string" #desc_expr})),
                "i32" | "i64" | "u32" | "u64" | "isize" | "usize" => {
                    quote!(serde_json::json!({"type": "integer" #desc_expr}))
                }
                "f32" | "f64" => quote!(serde_json::json!({"type": "number" #desc_expr})),
                "bool" => quote!(serde_json::json!({"type": "boolean" #desc_expr})),
                "Vec" => {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            let inner_schema = schema_expr(inner_ty, "");
                            quote!(
                                serde_json::json!({"type": "array", "items": #inner_schema #desc_expr})
                            )
                        } else {
                            quote!(serde_json::json!({"type": "array" #desc_expr}))
                        }
                    } else {
                        quote!(serde_json::json!({"type": "array" #desc_expr}))
                    }
                }
                "Option" => {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            schema_expr(inner_ty, desc)
                        } else {
                            quote!(serde_json::json!({"type": "string" #desc_expr}))
                        }
                    } else {
                        quote!(serde_json::json!({"type": "string" #desc_expr}))
                    }
                }
                _ => {
                    // Assume it's a ToolArg
                    let ty_ident = &seg.ident;
                    if desc.is_empty() {
                        quote!(<#ty_ident as ToolArg>::schema())
                    } else {
                        quote!({
                            let mut s = <#ty_ident as ToolArg>::schema();
                            s["description"] = serde_json::Value::String(#desc.to_string());
                            s
                        })
                    }
                }
            }
        } else {
            quote!(serde_json::json!({"type": "string" #desc_expr}))
        }
    } else {
        quote!(serde_json::json!({"type": "string" #desc_expr}))
    }
}

/// Checks if a type is Option<T>.
fn is_option(ty: &syn::Type) -> bool {
    if let syn::Type::Path(p) = ty {
        p.path
            .segments
            .last()
            .map(|s| s.ident == "Option")
            .unwrap_or(false)
    } else {
        false
    }
}

/// Converts snake_case to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}
