//! A minimal JSON writer for the machine readable reports.
//!
//! growforge writes one small, fixed-shape document (the stress report) and
//! reads none, so it carries the twenty lines that needs rather than a
//! serialization dependency.

/// Quote and escape a string as a JSON string literal, delimiters included.
///
/// Everything below `0x20` goes out as a `\u00xx` escape, which keeps the
/// output valid for any project name a user can type.
pub fn string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Render a number as a JSON literal. JSON has no infinity or NaN, so anything
/// that is not finite becomes `null`.
pub fn number(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_string()
    }
}

/// Render an optional number, with `None` becoming `null`.
pub fn optional_number(value: Option<f64>) -> String {
    match value {
        Some(v) => number(v),
        None => "null".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_quoted_and_escaped() {
        assert_eq!(string("plain"), "\"plain\"");
        assert_eq!(string("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(string("line\nbreak\ttab"), "\"line\\nbreak\\ttab\"");
        assert_eq!(string("\u{1}"), "\"\\u0001\"");
        // Anything printable passes through, including non-ASCII.
        assert_eq!(string("brücke"), "\"brücke\"");
    }

    #[test]
    fn numbers_round_trip_and_non_finite_becomes_null() {
        assert_eq!(number(1.5), "1.5");
        assert_eq!(number(0.0), "0");
        assert_eq!(number(-2.0), "-2");
        assert_eq!(number(f64::NAN), "null");
        assert_eq!(number(f64::INFINITY), "null");
        assert_eq!(optional_number(None), "null");
        assert_eq!(optional_number(Some(3.25)), "3.25");
        // The literal must parse back to the same double.
        let value = 1.234_567_890_123_456_7e-5;
        assert_eq!(number(value).parse::<f64>().unwrap(), value);
    }
}
