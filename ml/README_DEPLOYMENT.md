# ML-FORGE Streamlit Deployment Guide

This guide provides instructions for deploying the ML-FORGE application on Streamlit Cloud.

## Deployment Steps

1. **Create a GitHub Repository**
   - Create a new GitHub repository for this project
   - Push all the optimized files to this repository

2. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Log in with your GitHub account
   - Click "New app"
   - Select your repository, branch (main), and the main file (streamlit_app.py)
   - Click "Deploy"

## Project Structure

The project has been optimized with the following files to ensure smooth deployment:

- `streamlit_app.py` - Optimized entry point for Streamlit Cloud
- `app.py` - Main application code
- `requirements.txt` - Optimized dependencies with pinned versions
- `.streamlit/config.toml` - Streamlit configuration for better performance
- `Procfile` - Instructions for deployment platforms
- `runtime.txt` - Specifies Python version
- `setup.sh` - Setup script for deployment environment

## Troubleshooting

If you encounter deployment issues:

1. **Memory Errors**
   - Streamlit Cloud has memory limitations. The requirements.txt has been optimized to reduce memory usage.
   - Optional dependencies are commented out and can be added back if needed.

2. **Slow Startup**
   - The application uses lazy loading to improve startup time.
   - First load may still be slow due to dependency installation.

3. **Dependency Conflicts**
   - The requirements.txt file includes specific versions to avoid conflicts.
   - The protobuf dependency has been specifically set to resolve TensorFlow compatibility issues.

4. **Application Crashes**
   - Check the logs in Streamlit Cloud for detailed error messages.
   - You may need to increase the resources in your Streamlit Cloud settings.

## Local Testing

Before deploying to Streamlit Cloud, you can test locally:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## Additional Optimizations

If you still encounter deployment issues:

1. Consider using smaller ML models or model quantization.
2. Split the application into multiple pages using Streamlit's multipage app feature.
3. Use caching liberally with `@st.cache_data` and `@st.cache_resource` decorators.
4. Load data and models only when needed, not at application startup. 