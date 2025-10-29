from supabase import create_client, Client
from dotenv import load_dotenv
import os
import time

print("--- Starting Supabase connection test ---")

try:
    # 1. Load .env file
    load_dotenv()
    print("INFO: Loaded .env file.")

    # 2. Get credentials
    supabase_URL = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    # 3. Print credentials to check them (partially hiding the key)
    print(f"INFO: Loaded SUPABASE_URL: {supabase_URL}")
    print(f"INFO: Loaded SUPABASE_SERVICE_KEY: ...{supabase_key[-4:] if supabase_key else '!!! NOT FOUND !!!'}")

    if not supabase_URL or not supabase_key:
        print("\n[FATAL ERROR] SUPABASE_URL or SUPABASE_SERVICE_KEY is missing.")
        print("Please check your .env file.")
    else:
        # 4. Create the client
        print("INFO: Creating Supabase client...")
        supabase: Client = create_client(supabase_URL, supabase_key)
        print("INFO: Client object created (this does not mean it's connected yet).")

        # 5. Force a connection by making a real request
        print("INFO: Attempting to connect and list storage buckets...")
        start_time = time.time()
        
        # This is the line that will actually test the connection and auth
        buckets = supabase.storage.list_buckets()
        
        end_time = time.time()
        print(f"\n--- SUCCESS! ---")
        print(f"Connected and authenticated successfully in {end_time - start_time:.2f} seconds.")
        print("Available buckets in your project:")
        if buckets:
            for bucket in buckets:
                print(f"- {bucket.name}")
        else:
            print("- No buckets found.")

except Exception as e:
    print(f"\n--- CONNECTION FAILED ---")
    print(f"An error occurred: {e}")

print("\n--- Test script finished ---")
