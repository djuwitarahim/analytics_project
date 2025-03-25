import os
import shutil

def create_project_structure():
    """Create the project structure for the Online Shopper Intention Predictor application"""
    print("Setting up project structure for Online Shopper Intention Predictor...")
    
    # Create main directories
    os.makedirs("models", exist_ok=True)
    print("✅ Created 'models' directory")
    
    # Create empty __init__.py files to make directories importable
    open(os.path.join("models", "__init__.py"), "w").close()
    print("✅ Created __init__.py files")
    
    print("\nProject structure created successfully!")
    print("\nTo run the application, use the following command:")
    print("streamlit run app.py")

if __name__ == "__main__":
    create_project_structure()