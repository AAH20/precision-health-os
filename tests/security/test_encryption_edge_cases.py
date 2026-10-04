"""Security tests for encryption edge cases."""

import os

import pytest

from precision_health_os.security import EncryptionService, KeyRotationService


class TestEncryptionService:
    """Tests for AES-256-GCM encryption edge cases."""

    def setup_method(self):
        self.key = os.urandom(32)
        self.service = EncryptionService(self.key)

    def test_encrypt_decrypt_empty_string(self):
        """Encrypting and decrypting an empty string should work."""
        plaintext = ""
        ciphertext = self.service.encrypt(plaintext)
        assert self.service.decrypt(ciphertext) == plaintext

    def test_encrypt_decrypt_large_string(self):
        """Encrypting and decrypting a 1MB string should work."""
        plaintext = "A" * (1024 * 1024)
        ciphertext = self.service.encrypt(plaintext)
        assert self.service.decrypt(ciphertext) == plaintext

    def test_encrypt_same_text_twice_produces_different_ciphertexts(self):
        """Encrypting the same plaintext twice should produce different ciphertexts."""
        plaintext = "Hello, World!"
        ct1 = self.service.encrypt(plaintext)
        ct2 = self.service.encrypt(plaintext)
        assert ct1 != ct2

    def test_tampered_ciphertext_raises(self):
        """Tampering with the ciphertext should cause decryption to raise."""
        plaintext = "Sensitive data"
        ciphertext = bytearray(self.service.encrypt(plaintext))
        ciphertext[15] ^= 0x01
        from cryptography.exceptions import InvalidTag

        with pytest.raises(InvalidTag):
            self.service.decrypt(bytes(ciphertext))

    def test_tampered_associated_data_raises(self):
        """Tampering with associated data should cause decryption to raise."""
        plaintext = "Sensitive data"
        aad = b"original aad"
        ciphertext = self.service.encrypt(plaintext, aad)
        from cryptography.exceptions import InvalidTag

        with pytest.raises(InvalidTag):
            self.service.decrypt(ciphertext, b"tampered aad")

    def test_decrypt_with_wrong_key_raises(self):
        """Decrypting with the wrong key should raise."""
        plaintext = "Sensitive data"
        ciphertext = self.service.encrypt(plaintext)
        wrong_key = os.urandom(32)
        wrong_service = EncryptionService(wrong_key)
        from cryptography.exceptions import InvalidTag

        with pytest.raises(InvalidTag):
            wrong_service.decrypt(ciphertext)

    def test_key_must_be_32_bytes(self):
        """Key must be exactly 32 bytes for AES-256."""
        with pytest.raises(ValueError):
            EncryptionService(os.urandom(31))
        with pytest.raises(ValueError):
            EncryptionService(os.urandom(33))
        service = EncryptionService(os.urandom(32))
        assert service is not None


class TestKeyRotationService:
    """Tests for key rotation edge cases."""

    def test_rotate_generates_new_key(self):
        """Rotating should generate a new key ID."""
        krs = KeyRotationService()
        key_id1 = krs.generate_key()
        key_id2 = krs.rotate()
        assert key_id1 != key_id2
        assert krs.get_current_key_id() == key_id2

    def test_old_keys_can_still_decrypt_after_rotation(self):
        """Old keys should still be able to decrypt after rotation."""
        krs = KeyRotationService()
        old_key_id = krs.generate_key()
        old_key = krs.get_key(old_key_id)

        enc_service = EncryptionService(old_key)
        plaintext = "Data encrypted with old key"
        ciphertext = enc_service.encrypt(plaintext)

        new_key_id = krs.rotate()
        assert new_key_id != old_key_id

        assert enc_service.decrypt(ciphertext) == plaintext

    def test_get_current_key_id_raises_before_any_key(self):
        """get_current_key_id should raise before any key exists."""
        krs = KeyRotationService()
        with pytest.raises(RuntimeError):
            krs.get_current_key_id()
