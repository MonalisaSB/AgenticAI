import streamlit as st
import requests
from autogen import AssistantAgent  
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
weather_api_key = os.getenv("weather_api_key")

# Step 1: Define the function to fetch weather data using the OpenWeatherMap API
def get_weather(city: str, api_key: str) -> str:
    api_key = "9e0c9a2d7478f9d535424f1778fc6a9d"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        weather_description = data['weather'][0]['description']
        return f"The current temperature in {city} is {temperature}°C with {weather_description}."
    else:
        return "Unable to fetch weather data."

# Step 2: Define the Weather Agent using AutoGen
weather_agent = AssistantAgent(
    name="Weather_Agent",
    llm_config={
        "config_list": [
            {
                "api_type": "azure",
                "azure_endpoint": "https://openai-api-management-gw.azure-api.net/openaiprodtest/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview" ,
                "api_version": "2023-12-01-preview",
                "api_key": "2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",
                # "deployment_name": "gpt-4o-mini",  # corrected key name
                "model": "gpt-4o-mini",  # optional, but good to keep
            }
        ],
        "temperature": 0.7,
    },
    system_message="I can fetch weather data for any city.",
)

# Step 3: Streamlit UI
def main():
    st.title("Weather Information Agent")
    st.markdown("Enter the name of a city to get current weather information.")

    city = st.text_input("Enter City Name:")

    if city and st.button("Get Weather"):
        result = get_weather(city, weather_api_key)  # Fetch weather data

        # Display result
        st.write(result)

# Step 4: Run the Streamlit app
if __name__ == "__main__":
    main()