import openai
from openai import AzureOpenAI
from typing import List

class NeurodiverseAdapter:
    COMPRESSION_MAPPING = {
        "brief": 0.25,   # 1-2 sentences
        "short": 0.5,    # 2-4 sentences
        "medium": 0.75,  # 4-10 sentences
        "long": 0.9      # 10-15 sentences
    }

    def __init__(self, api_key: str, endpoint: str, api_version: str, model_name: str):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model_name = model_name
        
    def adapt_text(self, text: str, interests: List[str], compression_level: str) -> str:
        if compression_level not in self.COMPRESSION_MAPPING:
            raise ValueError(f"Compression level must be one of: {', '.join(self.COMPRESSION_MAPPING.keys())}")
            
        numerical_compression = self.COMPRESSION_MAPPING[compression_level]
        interests_str = ", ".join(interests[:5])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""Rephrase the following text using concepts from these interests: {interests_str}. 
                    Make it {compression_level} ({self.get_sentence_range(compression_level)}) and accessible for a neurodiverse audience: {text}"""}
        ]
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        
        return completion.choices[0].message.content.strip()

    @staticmethod
    def get_sentence_range(level: str) -> str:
        ranges = {
            "brief": "1-2 sentences",
            "short": "2-4 sentences",
            "medium": "4-10 sentences",
            "long": "10-15 sentences"
        }
        return ranges[level]

def get_api_key():
    with open('api.txt', 'r') as f:
        return f.read().strip()

def main():
    API_KEY = get_api_key()
    ENDPOINT = "https://mango-bush-0a9e12903.5.azurestaticapps.net/api/v1"
    API_VERSION = "2024-02-01"
    MODEL_NAME = "gpt-4o"
    adapter = NeurodiverseAdapter(API_KEY, ENDPOINT, API_VERSION, MODEL_NAME)
    
    # Example usage with new string labels
    result = adapter.adapt_text(
        "mitochondria is the powerhouse of the cell", 
        ["Football"], 
        "long"  # Use "brief", "short", "medium", or "long"
    )
    print("Adapted text:")
    print(result)

if __name__ == "__main__":
    main()