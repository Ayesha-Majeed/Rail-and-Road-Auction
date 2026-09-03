import os
import time
from backend.crypto_utils import CryptoUtils

class DBConnector:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False

    def connect(self):
        # Reduced retries and timeout for faster feedback when offline
        max_retries = 2
        last_error = ""
        
        for attempt in range(1, max_retries + 1):
            try:
                # pyrefly: ignore [import-unresolved, missing-import]
                from pymongo import MongoClient
                # Fast timeout: 3 seconds is enough for a healthy connection
                self.client = MongoClient(self.uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
                self.client.admin.command("ping")
                self.db = self.client[self.db_name]
                self.connected = True
                return True, f"Connected to MongoDB — DB: '{self.db_name}'"
            except Exception as e:
                # USER REQUEST: Handle network errors gracefully with a friendly message.
                detailed_err = str(e)
                if "topology" in detailed_err.lower() or "timeout" in detailed_err.lower() or "reachable" in detailed_err.lower():
                    last_error = "Network Error: Could not reach MongoDB. Please check your internet connection."
                else:
                    last_error = detailed_err
                
                if attempt < max_retries:
                    time.sleep(1.0) # Faster retry interval
                continue
        
        self.connected = False
        return False, last_error

    def book_exists(self, collection, book_id):
        try:
            return self.db[collection].find_one({"book_id": book_id}) is not None
        except:
            return False

    def book_title_exists(self, collection, new_title, return_doc=False):
        if not new_title: return None if return_doc else False
        if isinstance(new_title, list): new_title = " ".join(new_title)
        new_title = str(new_title)
        
        try:
            # Fetch all titles from the database to compare
            # Fetch full document if return_doc is True so we can use it
            projection = None if return_doc else {"title": 1}
            books = self.db[collection].find({}, projection)
            from thefuzz import fuzz
            new_title_lower = new_title.lower()
            print(f"\n🔍 Checking API/OCR title: '{new_title}' against DB titles...")
            for b in books:
                t = b.get("title", "")
                if not t: continue
                if isinstance(t, list): t = " ".join(t)
                t = str(t)
                
                similarity = fuzz.token_sort_ratio(new_title_lower, t.lower())
                if similarity >= 90:
                    print(f"   ↳ DB Title: '{t}' | Score: {similarity}%")
                    print(f"   ✅ Match found! (Score: {similarity}%) Skipping duplicate.")
                    return b if return_doc else True
            print("   ❌ No match found >= 90%. Book is unique.")
            return None if return_doc else False
        except Exception as e:
            print(f"❌ DB title check error: {e}")
            return None if return_doc else False

    def find_user_by_token(self, token: str):
        """
        Iterates through users and decrypts their stored tokens to find a match.
        The secret key is provided by the collaborator (Wasi Shah).
        """
        SECRET_KEY = os.environ.get("CONNECTION_SECRET_KEY", "78752a9db25d08be9e4702510374164335e63863aae30e8e212ac79a8884c354") # Fallback for local dev
        try:
            # 1. Get all users who have an encrypted token
            users = self.db["users"].find({"desktopConnectionTokenEnc": {"$exists": True}})
            for user in users:
                enc_token = user.get("desktopConnectionTokenEnc")
                if not enc_token: continue
                
                # 2. Attempt Decryption using verified strategy
                decrypted = CryptoUtils.decrypt_token(enc_token, SECRET_KEY)
                
                # 3. Compare with input
                if decrypted == token:
                    return user
            return None
        except Exception as e:
            print(f"❌ User lookup failed: {e}")
            return None

    def insert_book(self, collection, doc):
        result = self.db[collection].insert_one(doc)
        return str(result.inserted_id)

    def ping(self):
        try:
            if not self.client:
                return False
            self.client.admin.command("ping")
            self.connected = True
            return True
        except Exception:
            self.connected = False
            return False

    def disconnect(self):
        try:
            if self.client:
                self.client.close()
        except:
            pass
        self.connected = False
