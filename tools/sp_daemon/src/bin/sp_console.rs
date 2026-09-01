use std::convert::Infallible;
use std::time::Duration;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Json,
    },
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::info;

#[derive(Serialize)]
struct NodeTelemetry {
    node_id: String,
    cpu_temp_c: f32,
    svm_mem_gb: f32,
    dht_peers_active: u32,
    dht_peers_total: u32,
    pouw_frontier: u64,
}

async fn node_telemetry(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(telemetry_loop)
}

async fn telemetry_loop(mut socket: WebSocket) {
    loop {
        let payload = NodeTelemetry {
            node_id: "q3-beast-canyon".to_string(),
            cpu_temp_c: 58.5,
            svm_mem_gb: 2.4,
            dht_peers_active: 14,
            dht_peers_total: 32,
            pouw_frontier: 1_048_576,
        };
        let json = serde_json::to_string(&payload).expect("NodeTelemetry is always serializable");
        if socket.send(Message::Text(json)).await.is_err() {
            break;
        }
        sleep(Duration::from_millis(1000)).await;
    }
}

#[derive(Deserialize)]
struct ChatRequest {
    prompt: String,
}

async fn chat_handler(
    Json(req): Json<ChatRequest>,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(100);
    let prompt = req.prompt;

    tokio::spawn(async move {
        let response = format!(
            "System online. Beast Canyon math-core initialized. \
             Awaiting retrocausal QUIC streams. You said: {}",
            prompt
        );
        for token in response.split_whitespace() {
            let event = Ok(Event::default().data(token.to_owned()));
            if tx.send(event).await.is_err() {
                break;
            }
            sleep(Duration::from_millis(50)).await;
        }
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

async fn chat_stream_stub() -> Json<Value> {
    Json(serde_json::json!({"status": "stub", "stream": "sse-legacy"}))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let app = Router::new()
        .route("/v1/chat", post(chat_handler))
        .route("/v1/chat/stream", get(chat_stream_stub))
        .route("/v1/node/telemetry", get(node_telemetry))
        .fallback_service(ServeDir::new("frontend_mockups"))
        .layer(CorsLayer::permissive());

    let addr: std::net::SocketAddr = ([127, 0, 0, 1], 3000).into();
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind 127.0.0.1:3000");
    info!("operator console listening on {addr}");
    axum::serve(listener, app).await.expect("server error");
}
