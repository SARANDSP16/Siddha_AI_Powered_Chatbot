# Updated app.py

from streamlit import Streamlit
from rag import format_output  # Import the function

# Sample usage to demonstrate enhanced output

# Assuming there's a function in rag.py that provides data to be formatted
result = some_function()  # Placeholder for actual function
formatted_result = format_output(result)  # Use format_output

# Now display the formatted result on Streamlit
Streamlit.write(formatted_result)  # Updated display function
