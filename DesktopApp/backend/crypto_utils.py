class CryptoUtils:
    """
    Handles AES-256-GCM decryption for tokens encrypted by Node.js.
    Format expected: iv.at.ct (dot-separated Base64 segments)
    """
    @staticmethod
    def decrypt_token(encrypted_str: str, hex_key: str):
        if not encrypted_str or not hex_key:
            return None
        try:
            import base64
            import hashlib
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            # 1. Prepare Key (Verified strategy: SHA-256 of the secret string)
            key = hashlib.sha256(hex_key.encode('utf-8')).digest()
            
            # 2. Parse Segments (Node.js style: iv.tag.cipher)
            parts = encrypted_str.split('.')
            if len(parts) != 3:
                return None
            
            iv_b64, tag_b64, cipher_b64 = parts

            def b64_decode(s):
                # Add padding if needed
                s += '=' * (-len(s) % 4)
                # Try URL-safe first (standard for modern Node.js)
                try:
                    return base64.urlsafe_b64decode(s)
                except:
                    return base64.b64decode(s)

            iv = b64_decode(iv_b64)
            tag = b64_decode(tag_b64)
            ciphertext = b64_decode(cipher_b64)

            # 3. Decrypt (AESGCM expects cipher + tag combined)
            aesgcm = AESGCM(key)
            decrypted = aesgcm.decrypt(iv, ciphertext + tag, None)
            return decrypted.decode('utf-8')
        except Exception as e:
            # Silent fail for lookup iteration
            return None
