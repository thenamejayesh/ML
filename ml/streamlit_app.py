import streamlit as st
import os
import sys

# Configure page with minimal settings
st.set_page_config(
    page_title="ML-FORGE",
    page_icon="🔧",
    layout="wide"
)

# Create a loading message
st.write("# ML-FORGE")
with st.spinner("Loading application... This may take a moment."):
    try:
        # Add the project root to sys.path
        root_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, root_dir)
        
        # Ensure the src directory is in sys.path
        src_dir = os.path.join(root_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        
        # Import the main app module
        from app import main
        
        # Run the main function
        main()
        
    except Exception as e:
        st.error(f"Error loading the application: {str(e)}")
        st.info("Please check the logs for more details or try again later.")
        
        # Show details in expander
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())
