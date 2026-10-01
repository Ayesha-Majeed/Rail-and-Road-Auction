import os
import re
import time
from backend.crypto_utils import CryptoUtils

# Configure fast, reliable public DNS servers (Google 8.8.8.8, Cloudflare 1.1.1.1)
# to completely bypass slow / timing-out local router DNS (192.168.0.1) on Linux
try:
    import dns.resolver
    _default_res = dns.resolver.get_default_resolver()
    _default_res.nameservers = ['8.8.8.8', '1.1.1.1', '192.168.0.1']
    _default_res.lifetime = 6.0
    _default_res.timeout = 3.0
except Exception:
    pass

ROMAN_MAP = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10",
    "xi": "11", "xii": "12", "xiii": "13", "xiv": "14", "xv": "15"
}

def normalize_isbn(val):
    """Normalizes an ISBN by removing hyphens/spaces and validating length."""
    if not val:
        return ""
    s = str(val).strip().upper().replace("-", "").replace(" ", "")
    if s in ("N/A", "NONE", "NULL", "NOTFOUND", "NOTAVAILABLE"):
        return ""
    cleaned = "".join(c for c in s if c.isalnum())
    return cleaned if len(cleaned) >= 8 else ""

def isbns_match(isbn1, isbn2):
    """
    Returns:
      True  -> Both ISBNs are valid and match (same book)
      False -> Both ISBNs are valid but differ (different books)
      None  -> At least one ISBN is missing / invalid
    """
    i1 = normalize_isbn(isbn1)
    i2 = normalize_isbn(isbn2)
    if not i1 or not i2:
        return None
    if i1 == i2:
        return True
    # Handle ISBN-10 vs ISBN-13 (prefix 978)
    if len(i1) == 10 and len(i2) == 13 and i2.startswith("978") and i2[3:12] == i1[:9]:
        return True
    if len(i2) == 10 and len(i1) == 13 and i1.startswith("978") and i1[3:12] == i2[:9]:
        return True
    return False

def extract_volume(*texts):
    """
    Extracts volume identifier (e.g. '1', '2', '3', 's') from title, subtitle, or edition.
    """
    for text in texts:
        if not text:
            continue
        text = str(text).strip()
        # 1. Matches "Volume 2", "Vol. 2", "V. 2", "Volume S"
        m = re.search(r"\b(?:volume|vol\.?|v\.)\s*([0-9ivxldcm]+|[a-z])\b", text, re.IGNORECASE)
        if m:
            val = m.group(1).lower()
            return ROMAN_MAP.get(val, val)
        # 2. Matches "Part 1", "Pt. 2", "Book 3", "Bk 2"
        m2 = re.search(r"\b(?:part|pt\.?|book|bk\.?)\s*([0-9ivxldcm]+|[a-d])\b", text, re.IGNORECASE)
        if m2:
            val = m2.group(1).lower()
            return ROMAN_MAP.get(val, val)
    return None

def clean_base_title(text):
    """
    Strips volume / part indicators from the title to compare the pure series / work title.
    """
    if not text:
        return ""
    text = str(text).strip()
    # Strip trailing volume info like ", Volume 2", " - Vol. 3", " Part I", " Volume S"
    cleaned = re.sub(r"[\s,\-\:]+\b(?:volume|vol\.?|v\.|part|pt\.?|book|bk\.?)\s*([0-9ivxldcm]+|[a-z])\b.*$", "", text, flags=re.IGNORECASE)
    # Strip leading volume info if any
    cleaned = re.sub(r"^\b(?:volume|vol\.?|v\.|part|pt\.?|book|bk\.?)\s*([0-9ivxldcm]+|[a-z])[\s,\-\:]+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def check_same_book_match(new_title, new_isbn, db_title, db_isbn):
    """
    Determines if two records with the same book_id represent the exact same book or different books.
    
    Rules:
    1. Title must match (similarity >= 70%).
       If titles differ (< 70%) -> Different books (e.g. lot number reused in future auction).
    2. When titles match:
       - If both have valid ISBNs:
           - Same ISBN  -> True (Same book / duplicate)
           - Differ ISBN -> False (Different books / save both)
       - If one has ISBN and the other does not:
           - False (Different books / save both)
       - If neither book has an ISBN:
           - True (Same book / duplicate)
    """
    if not new_title or not db_title:
        return False, 0, "Missing title in one or both records"

    from thefuzz import fuzz
    new_base = clean_base_title(new_title)
    db_base = clean_base_title(db_title)
    base_sim = fuzz.token_sort_ratio(new_base.lower(), db_base.lower())
    raw_sim = fuzz.token_sort_ratio(str(new_title).lower(), str(db_title).lower())
    match_score = max(base_sim, raw_sim)

    if match_score < 70:
        return False, match_score, f"Titles do not match ({match_score}% < 70%)"

    # Volume check: Different volumes of the same series/title are DIFFERENT books
    vol_new = extract_volume(new_title)
    vol_db = extract_volume(db_title)
    if vol_new is not None and vol_db is not None and vol_new != vol_db:
        return False, match_score, f"Titles belong to different volumes ('{vol_new}' vs '{vol_db}') — Not a duplicate"

    norm_new = normalize_isbn(new_isbn)
    norm_db = normalize_isbn(db_isbn)

    if norm_new and norm_db:
        isbn_same = isbns_match(norm_new, norm_db)
        if isbn_same is True:
            return True, match_score, f"Titles match ({match_score}%) and ISBNs match ({norm_new})"
        else:
            return False, match_score, f"Titles match ({match_score}%) but ISBNs differ ({norm_new} vs {norm_db})"
    elif norm_new and not norm_db:
        return False, match_score, f"Titles match ({match_score}%) but only new record has ISBN ({norm_new})"
    elif not norm_new and norm_db:
        return False, match_score, f"Titles match ({match_score}%) but only DB record has ISBN ({norm_db})"
    else:
        # Neither has ISBN
        return True, match_score, f"Titles match ({match_score}%) and neither record has an ISBN"

class DBConnector:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False

    def connect(self):
        # 3 retries with 6-second timeout for stable connection & DNS resolution
        max_retries = 3
        last_error = ""
        
        for attempt in range(1, max_retries + 1):
            try:
                # pyrefly: ignore [import-unresolved, missing-import]
                from pymongo import MongoClient
                mongo_kwargs = {
                    "serverSelectionTimeoutMS": 6000,
                    "connectTimeoutMS": 6000,
                    "socketTimeoutMS": 6000
                }
                try:
                    import certifi
                    mongo_kwargs["tlsCAFile"] = certifi.where()
                except Exception:
                    pass

                self.client = MongoClient(self.uri, **mongo_kwargs)
                self.client.admin.command("ping")
                self.db = self.client[self.db_name]
                self.connected = True
                self._ensure_indexes()
                return True, f"Connected to MongoDB — DB: '{self.db_name}'"
            except Exception as e:
                detailed_err = str(e)
                if any(k in detailed_err.lower() for k in ["topology", "timeout", "reachable", "name resolution", "errno -3", "resolution lifetime"]):
                    last_error = f"Network Error: Could not reach MongoDB ({e}). Please check your internet connection."
                else:
                    last_error = detailed_err
                
                if attempt < max_retries:
                    time.sleep(1.0 * attempt)
                continue
        
        self.connected = False
        return False, last_error

    def _ensure_indexes(self):
        """Ensures high-performance indexes on MongoDB collections."""
        try:
            if self.db is not None:
                self.db["Book Data"].create_index("cover_sha256", sparse=True)
                self.db["Book Data"].create_index("book_id", sparse=True)
        except Exception:
            pass

    def reconnect(self, silent=False):
        """Attempts to re-establish the MongoDB connection after a transient disconnect/DNS drop."""
        try:
            if not silent:
                print("🔄 Re-establishing MongoDB connection...")
            from pymongo import MongoClient
            mongo_kwargs = {
                "serverSelectionTimeoutMS": 6000,
                "connectTimeoutMS": 6000,
                "socketTimeoutMS": 6000
            }
            try:
                import certifi
                mongo_kwargs["tlsCAFile"] = certifi.where()
            except Exception:
                pass

            new_client = MongoClient(self.uri, **mongo_kwargs)
            new_client.admin.command("ping")
            old_client = self.client
            self.client = new_client
            self.db = self.client[self.db_name]
            self.connected = True
            self._ensure_indexes()
            if old_client:
                try:
                    old_client.close()
                except Exception:
                    pass
            if not silent:
                print("✅ Reconnected to MongoDB successfully.")
            return True
        except Exception as e:
            self.connected = False
            if not silent:
                print(f"⚠️ Reconnection attempt failed: {e}")
            return False

    def _execute_with_retry(self, operation_fn, op_name="DB operation", max_retries=3, initial_delay=1.0):
        """
        Executes a database callable with automatic retry and reconnection
        upon transient network / DNS errors (e.g. Errno -3, AutoReconnect, timeouts).
        """
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                if not self.connected or self.client is None or self.db is None:
                    if not self.reconnect(silent=False):
                        raise RuntimeError("MongoDB is currently unreachable due to network/DNS timeout.")
                return operation_fn()
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                is_transient = any(term in err_str for term in [
                    "temporary failure in name resolution", "autoreconnect",
                    "serverselectiontimeouterror", "networktimeout", "timeout",
                    "connection refused", "broken pipe", "connection reset",
                    "errno -3", "nodename nor servname provided", "socket error",
                    "resolution lifetime expired", "dns operation timed out",
                    "cannot use mongoclient after close", "unreachable"
                ])
                if is_transient and attempt < max_retries:
                    wait_time = initial_delay * attempt
                    print(f"⚠️ {op_name} hit transient error: {e}. Retrying ({attempt}/{max_retries}) in {wait_time}s...")
                    time.sleep(wait_time)
                    self.reconnect(silent=False)
                    continue
                else:
                    raise e
        if last_exception:
            raise last_exception

    def book_exists(self, collection, book_id):
        def _check():
            return self.db[collection].find_one({"book_id": book_id}) is not None
        try:
            return self._execute_with_retry(_check, op_name=f"book_exists({book_id})")
        except Exception as e:
            print(f"❌ book_exists error: {e}")
            return False

    def update_book_sync_date(self, collection, doc_id_or_doc):
        """
        Updates the sync_date / synced_at timestamp of an existing book document in MongoDB.
        """
        if not doc_id_or_doc:
            return False
        
        try:
            from datetime import datetime
            now_iso = datetime.now().isoformat()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            query = {}
            if isinstance(doc_id_or_doc, dict):
                # Avoid redundant network round-trip if book was synced within the last 5 minutes
                last_s = doc_id_or_doc.get("last_synced") or doc_id_or_doc.get("synced_at")
                if last_s:
                    try:
                        prev_time = datetime.fromisoformat(str(last_s).replace("Z", "")) if "T" in str(last_s) else datetime.strptime(str(last_s), "%Y-%m-%d %H:%M:%S")
                        if (datetime.now() - prev_time).total_seconds() < 300:
                            return True
                    except Exception:
                        pass

                if "_id" in doc_id_or_doc:
                    query = {"_id": doc_id_or_doc["_id"]}
                elif "book_id" in doc_id_or_doc:
                    query = {"book_id": doc_id_or_doc["book_id"]}
                elif "title" in doc_id_or_doc:
                    query = {"title": doc_id_or_doc["title"]}
            else:
                query = {"_id": doc_id_or_doc}

            if not query:
                return False
                
            update_payload = {
                "$set": {
                    "synced_at": now_iso,
                    "sync_date": now_iso,
                    "last_synced": now_str,
                    "updated_at": now_iso
                }
            }
            
            def _update():
                return self.db[collection].update_one(query, update_payload)

            self._execute_with_retry(_update, op_name="update_book_sync_date")
            if isinstance(doc_id_or_doc, dict):
                doc_id_or_doc["synced_at"] = now_iso
                doc_id_or_doc["sync_date"] = now_iso
                doc_id_or_doc["last_synced"] = now_str
                doc_id_or_doc["updated_at"] = now_iso
            return True
        except Exception as e:
            print(f"❌ Failed to update book sync date: {e}")
            return False

    def update_book_fields(self, collection, doc_or_book_id, update_fields, user_id=None):
        """
        Updates specific fields (e.g. title, subtitle, author) of a book document in MongoDB.
        Targeting prioritizes _id if available, then book_id + user_id, with automatic retries.
        """
        if not doc_or_book_id or not update_fields:
            return False
            
        try:
            from datetime import datetime
            now_iso = datetime.now().isoformat()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = {"$set": dict(update_fields)}
            payload["$set"]["updated_at"] = now_iso
            payload["$set"]["last_synced"] = now_str
            
            query = {}
            if isinstance(doc_or_book_id, dict):
                if "_id" in doc_or_book_id and doc_or_book_id["_id"]:
                    query = {"_id": doc_or_book_id["_id"]}
                elif "book_id" in doc_or_book_id:
                    query = {"book_id": str(doc_or_book_id["book_id"]).strip()}
            else:
                from bson.objectid import ObjectId
                val_str = str(doc_or_book_id).strip()
                if ObjectId.is_valid(val_str):
                    query = {"_id": ObjectId(val_str)}
                else:
                    query = {"book_id": val_str}
            
            if user_id and "_id" not in query:
                from bson.objectid import ObjectId
                uids = [str(user_id)]
                if ObjectId.is_valid(str(user_id)):
                    uids.append(ObjectId(str(user_id)))
                query["user_id"] = {"$in": uids}
                
            def _update():
                res = self.db[collection].update_one(query, payload)
                return res.matched_count > 0 or res.modified_count > 0
                
            success = self._execute_with_retry(_update, op_name="update_book_fields")
            
            # Keep in-memory dict in sync
            if isinstance(doc_or_book_id, dict):
                for k, v in update_fields.items():
                    doc_or_book_id[k] = v
                doc_or_book_id["updated_at"] = now_iso
                doc_or_book_id["last_synced"] = now_str
                
            return success
        except Exception as e:
            print(f"❌ Failed to update book fields: {e}")
            return False

    def book_title_exists(
        self,
        collection,
        new_title,
        return_doc=False,
        update_sync_date=False,
        user_id=None,
        new_isbn=None,
        new_book_id=None,
        new_edition=None,
        new_subtitle=None
    ):
        if not new_title:
            return None if return_doc else False
        if isinstance(new_title, list):
            new_title = " ".join(new_title)
        new_title = str(new_title).strip()
        
        try:
            # Build filter query based on user_id if provided
            query = {}
            if user_id:
                from bson.objectid import ObjectId
                query_user_ids = [str(user_id)]
                if ObjectId.is_valid(str(user_id)):
                    query_user_ids.append(ObjectId(str(user_id)))
                query["user_id"] = {"$in": query_user_ids}

            # Fetch relevant fields for duplicate/volume/ISBN comparison
            projection = None if (return_doc or update_sync_date) else {
                "title": 1, "subtitle": 1, "edition": 1, "isbn": 1, "book_id": 1, "user_id": 1
            }

            def _fetch_books():
                return list(self.db[collection].find(query, projection))

            books = self._execute_with_retry(_fetch_books, op_name="fetch titles for duplicate check")
            from thefuzz import fuzz

            new_base = clean_base_title(new_title)
            vol_new = extract_volume(new_title, new_subtitle, new_edition)
            norm_new_isbn = normalize_isbn(new_isbn)

            print(f"\n🔍 Checking Book: '{new_title}' (Lot: {new_book_id}, ISBN: {new_isbn or 'None'}, Vol: {vol_new}) against DB...")

            for b in books:
                t = b.get("title", "")
                if not t:
                    continue
                if isinstance(t, list):
                    t = " ".join(t)
                t = str(t).strip()

                db_isbn = b.get("isbn")
                db_book_id = b.get("book_id")

                # ── RULE 1: AUCTION LOT NUMBER ISOLATION ──
                # In an auction, each Lot Number (e.g. 2079 vs 2080) is an independent item being sold.
                # Even if two books have the exact same title/author (e.g. multiple copies in the same auction),
                # each lot number MUST have its own record. Never skip another lot number!
                if new_book_id is not None and db_book_id is not None:
                    is_same_lot = (str(new_book_id).strip() == str(db_book_id).strip())
                    if not is_same_lot:
                        continue

                # ── RULE 2: MATCHING SAME LOT NUMBER (OR GENERAL DEDUPLICATION) ──
                db_base = clean_base_title(t)
                db_vol = extract_volume(t, b.get("subtitle"), b.get("edition"))

                base_sim = fuzz.token_sort_ratio(new_base.lower(), db_base.lower())
                raw_sim = fuzz.token_sort_ratio(new_title.lower(), t.lower())
                match_score = max(base_sim, raw_sim)

                isbn_match_status = isbns_match(new_isbn, db_isbn)

                # When checking against the SAME lot number:
                if new_book_id is not None and db_book_id is not None:
                    is_same, score, reason = check_same_book_match(new_title, new_isbn, t, db_isbn)
                    if is_same:
                        print(f"   ↳ DB: '{t}' | Score: {score}% | Book ID {new_book_id} | {reason}")
                        print(f"   ✅ Duplicate confirmed (Re-scan of same book)! Skipping.")
                        if update_sync_date:
                            self.update_book_sync_date(collection, b)
                        return b if return_doc else True
                    else:
                        print(f"   ↳ DB: '{t}' vs New: '{new_title}' | Book ID {new_book_id} | {reason} -> NOT a duplicate.")
                        continue

                # Fallback: When new_book_id was not provided (general deduplication)
                if match_score < 85:
                    continue

                if vol_new is not None and db_vol is not None:
                    if vol_new != db_vol:
                        continue
                    if isbn_match_status is False:
                        continue
                    print(f"   ↳ DB: '{t}' | Score: {match_score}% | Same Volume: {vol_new}")
                    if update_sync_date:
                        self.update_book_sync_date(collection, b)
                    return b if return_doc else True

                elif (vol_new is not None) != (db_vol is not None):
                    if isbn_match_status is True:
                        if update_sync_date:
                            self.update_book_sync_date(collection, b)
                        return b if return_doc else True
                    continue

                else:
                    if isbn_match_status is True:
                        if update_sync_date:
                            self.update_book_sync_date(collection, b)
                        return b if return_doc else True
                    elif isbn_match_status is False:
                        continue
                    else:
                        if match_score >= 90:
                            if update_sync_date:
                                self.update_book_sync_date(collection, b)
                            return b if return_doc else True

            print("   ❌ No duplicate match found. Book is unique.")
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
            def _fetch_users():
                return list(self.db["users"].find({"desktopConnectionTokenEnc": {"$exists": True}}))

            users = self._execute_with_retry(_fetch_users, op_name="find_user_by_token")
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
        def _insert():
            # Standardize all date fields so CSV/export filters never miss any records
            from datetime import datetime
            now_iso = datetime.now().isoformat()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if "synced_at" not in doc or not doc["synced_at"]:
                doc["synced_at"] = now_iso
            if "sync_date" not in doc or not doc["sync_date"]:
                doc["sync_date"] = now_iso
            if "last_synced" not in doc or not doc["last_synced"]:
                doc["last_synced"] = now_str
            if "upload_time" not in doc or not doc["upload_time"]:
                doc["upload_time"] = now_str
            if "created_at" not in doc or not doc["created_at"]:
                doc["created_at"] = now_iso

            # Auto-populate cover_sha256 if not already present
            if "cover_sha256" not in doc or not doc["cover_sha256"]:
                fc = doc.get("front_cover")
                if isinstance(fc, dict) and fc.get("file_path"):
                    c_hash = CryptoUtils.compute_file_sha256(fc.get("file_path"))
                    if c_hash:
                        doc["cover_sha256"] = c_hash
                        doc["front_cover"]["sha256"] = c_hash

            bid = doc.get("book_id")
            uid = doc.get("user_id")
            if bid:
                q = {"book_id": str(bid).strip()}
                if uid:
                    from bson.objectid import ObjectId
                    uids = [str(uid)]
                    if ObjectId.is_valid(str(uid)):
                        uids.append(ObjectId(str(uid)))
                    q["user_id"] = {"$in": uids}
                
                # Check if this book_id already exists in DB for this user
                existing_docs = list(self.db[collection].find(q))
                if existing_docs:
                    for ex in existing_docs:
                        ex_t = str(ex.get("title", "")).strip()
                        doc_t = str(doc.get("title", "")).strip()
                        is_same, score, reason = check_same_book_match(doc_t, doc.get("isbn"), ex_t, ex.get("isbn"))
                        
                        # If same book (title match + ISBN rules match) -> update existing document to prevent duplicates!
                        if is_same:
                            self.db[collection].update_one({"_id": ex["_id"]}, {"$set": doc})
                            print(f"   🔄 Updated existing document in DB for Book ID {bid} (Doc ID: {ex['_id']}) - {reason}")
                            return str(ex["_id"])

            result = self.db[collection].insert_one(doc)
            hash_display = doc.get("cover_sha256")
            hash_str = f" | SHA-256: {hash_display[:16]}..." if hash_display else ""
            print(f"   💾 Inserted new book into DB: Book ID {bid} (Doc ID: {result.inserted_id}){hash_str}")
            return str(result.inserted_id)
        return self._execute_with_retry(_insert, op_name=f"insert_book into {collection}")

    def find_by_cover_hash(self, collection, cover_hash, user_id=None, book_id=None):
        """
        Fast O(1) lookup of a book by its front cover SHA-256 hash.
        Returns: document dict if found, else None
        """
        if not cover_hash:
            return None
        def _find():
            query = {"cover_sha256": str(cover_hash).strip()}
            if user_id:
                from bson.objectid import ObjectId
                uids = [str(user_id)]
                if ObjectId.is_valid(str(user_id)):
                    uids.append(ObjectId(str(user_id)))
                query["user_id"] = {"$in": uids}

            # If book_id is provided, prioritize matching exact book_id first
            if book_id is not None:
                exact = self.db[collection].find_one({**query, "book_id": str(book_id).strip()})
                if exact:
                    return exact

            return self.db[collection].find_one(query)

        return self._execute_with_retry(_find, op_name="find_by_cover_hash")

    def ping(self):
        try:
            if not self.client:
                return False
            self.client.admin.command("ping")
            self.connected = True
            return True
        except Exception:
            # Try a quick silent reconnect before failing
            try:
                if self.reconnect(silent=True):
                    return True
            except Exception:
                pass
            self.connected = False
            return False

    def disconnect(self):
        try:
            if self.client:
                self.client.close()
        except:
            pass
        self.connected = False

