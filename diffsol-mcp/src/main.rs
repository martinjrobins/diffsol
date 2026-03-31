mod config;
mod error;
mod problem;
mod server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "diffsol_mcp=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    server::run_stdio().await
}
