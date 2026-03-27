# Build stage
FROM rust:1.86-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

# Runtime stage
FROM alpine:3.19

WORKDIR /app

COPY --from=builder /app/target/release/mlsys ./mlsys

ENTRYPOINT ["./mlsys"]
