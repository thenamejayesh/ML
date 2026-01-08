import os
import sys

def test_gemini_api():
    """Test if the Gemini API is working with the provided API key"""
    
    # Google Gemini API Key from your app
    gemini_api_key = "AIzaSyDIrW9Nmu9QeWPsL7YupStc2LP55o_gfuM"
    
    print("Starting Gemini API test...")
    print(f"API Key: {gemini_api_key[:5]}...{gemini_api_key[-5:]}")
    
    # Try with the new SDK first
    try:
        print("Attempting with new SDK (google.genai)...")
        from google import genai
        
        # Using new Google GenAI SDK
        client = genai.Client(api_key=gemini_api_key)
        
        # Simple test prompt
        test_prompt = "What is machine learning?"
        print(f"Sending test prompt: '{test_prompt}'")
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=test_prompt
        )
        
        # Print the response
        print("Response received successfully!")
        print("Response content:")
        print("-" * 40)
        print(response.text)
        print("-" * 40)
        print("New SDK test successful!")
        return True
        
    except ImportError:
        print("New SDK not available. Trying older SDK...")
    except Exception as e:
        print(f"Error with new SDK: {str(e)}")
        print("Trying older SDK as fallback...")
    
    # Try with the older SDK
    try:
        print("Attempting with older SDK (google.generativeai)...")
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=gemini_api_key)
        
        # Create a model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test prompt
        test_prompt = "What is machine learning?"
        print(f"Sending test prompt: '{test_prompt}'")
        
        response = model.generate_content(test_prompt)
        
        # Print the response
        print("Response received successfully!")
        print("Response content:")
        print("-" * 40)
        print(response.text)
        print("-" * 40)
        print("Older SDK test successful!")
        return True
        
    except ImportError:
        print("Older SDK not available.")
        print("Please install the required packages with:")
        print("pip install google-generativeai")
        return False
    except Exception as e:
        print(f"Error with older SDK: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_api() 