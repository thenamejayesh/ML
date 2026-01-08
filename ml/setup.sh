#!/bin/bash

# Create required directories
mkdir -p ~/.streamlit

# Write config
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false

[browser]
serverAddress = "0.0.0.0"
EOF

echo "Setup completed." 