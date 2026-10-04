# Security & Compliance Posture

**Scope.** This document describes the security controls actually implemented in
`src/precision_health_os/security/__init__.py`, as exercised by
`tests/security/test_security.py` and `tests/security/test_security_gaps.py`, and
the way audit/RBAC are wired in `src/precision_health_os/api/__init__.py`.
Every claim below is traceable to one of those files. Where a control is absent,
that absence is stated explicitly — see §6.

## 1. Encryption (EncryptionService)

`EncryptionService` (`security/__init__.py:22`) provides symmetric encryption for
PHI/PII using **AES-256-GCM** via `cryptography.hazmat.primitives.ciphers.aead.AESGCM`.

- **Key.** The constructor requires a 32-byte master key; any other length raises
  `ValueError` (`security/__init__.py:27-28`).
- **Nonce.** Each `encrypt()` call draws a fresh **96-bit (12-byte) random nonce**
  from `os.urandom(12)` (`security/__init__.py:33`). The nonce is prepended to the
  ciphertext (`nonce + ciphertext`, `security/__init__.py:36`) and split off again
  on decrypt (`security/__init__.py:40`). Because the nonce is random per message,
  encrypting the same plaintext twice yields different ciphertexts (asserted by
  `test_different_ciphertexts`, `test_security.py:35-39`).
- **Associated data.** `encrypt()`/`decrypt()` accept an optional `associated_data`
  bytes parameter (`security/__init__.py:31,38`). It is bound into the GCM
  authentication tag but not transmitted; callers must supply the same value on
  both sides. It is not used anywhere in the reviewed code.
- **Authentication guarantee.** GCM is an AEAD mode: the ciphertext is
  authenticated. Any modification of the nonce or ciphertext causes
  `AESGCM.decrypt` to raise `cryptography.exceptions.InvalidTag`
  (`security/__init__.py:42`). This is verified by
  `test_decrypt_tampered_ciphertext_raises` and
  `test_decrypt_tampered_nonce_raises` (`test_security_gaps.py:39-52`).
- **Convenience methods.** `encrypt_dict`/`decrypt_dict` JSON-encode/decode before
  encryption (`security/__init__.py:45-51`).

## 2. Key management (KeyRotationService)

`KeyRotationService` (`security/__init__.py:54`) manages encryption key rotation.

- `generate_key()` creates a new 256-bit key (`AESGCM.generate_key(bit_length=256)`),
  stores it under a generated id, and makes it current (`security/__init__.py:62-68`).
- **Rotation keeps old keys.** `rotate()` simply calls `generate_key()`
  (`security/__init__.py:82-84`). Previous keys remain in the `_keys` map, so
  ciphertext written under an older key id can still be decrypted by retrieving
  that key. There is no key deletion or expiry.
- **Unknown key id.** `get_key()` raises `KeyError` for an id not in the map
  (`security/__init__.py:72-73`; asserted by
  `test_get_key_unknown_id_raises_keyerror`, `test_security_gaps.py:58-61`).
- **No key yet.** `get_current_key_id()` raises `RuntimeError` if called before
  any key has been generated (`security/__init__.py:78-79`; asserted by
  `test_get_current_key_id_before_any_key_raises`, `test_security_gaps.py:67-70`).

## 3. Audit trail (AuditTrail)

`AuditTrail` (`security/__init__.py:87`) records HIPAA-style audit events with a
SHA-256 hash chain.

- **Chain construction.** Each `log()` call serializes
  `"{id}:{timestamp.isoformat()}:{user_id}:{action}:{resource_id}"` and computes
  `chain_hash = SHA256(f"{prev_hash}:{event_data}")`
  (`security/__init__.py:119-120`). The chain starts from `"0" * 64`
  (`security/__init__.py:93`).
- **Tamper detection.** `verify_chain()` recomputes the chain from the stored
  events and compares it to the running hash (`security/__init__.py:125-134`).
  If any logged field is altered after the fact, the recomputed hash differs and
  verification returns `False` (asserted by `test_tamper_detection`,
  `test_security.py:84-89`).
- **Querying.** `get_events()` filters by `user_id`, `resource_type`, and
  `start`/`end` timestamps (`security/__init__.py:136-153`).

**Honest limitation.** The chain is **in-process only**: both the event list and
the running hash live in memory on the `AuditTrail` instance
(`security/__init__.py:92-93`). An attacker who can rewrite the entire `_events`
list can recompute a valid `_chain_hash` and the tamper will not be detected.
`verify_chain()` detects *in-place* modification of an existing event, not
wholesale replacement. Real tamper-evidence requires **external anchoring** —
e.g. periodic export to write-once storage, signed timestamps, or a third-party
log service — none of which is implemented.

**API wiring.** `PrecisionHealthAPI` instantiates `AuditTrail` and `RBACService`
(`api/__init__.py:30-31`). In the reviewed code, audit events are written only by
`acknowledge_alert()` (`api/__init__.py:74`); `register_patient`, `get_patient`,
and `ingest_vitals` do not emit audit events. `check_permission()` delegates to
`RBACService.has_permission()` (`api/__init__.py:77-79`).

## 4. RBAC (RBACService)

`RBACService` (`security/__init__.py:156`) defines five roles:

| Role | Permissions |
|---|---|
| `admin` | `read`, `write`, `delete`, `admin`, `audit` |
| `physician` | `read`, `write`, `prescribe`, `order` |
| `nurse` | `read`, `write`, `administer` |
| `researcher` | `read`, `export`, `anonymize` |
| `patient` | `read_own`, `write_own` |

(`security/__init__.py:159-165`.)

- `assign_role()` raises `ValueError` for an unknown role
  (`security/__init__.py:173-174`).
- **Permissions are unioned across roles.** A user with multiple roles has the
  union of all their roles' permissions (`get_user_permissions`,
  `security/__init__.py:182-188`; `has_permission` checks any role,
  `security/__init__.py:177-180`). A user with no roles has an empty permission
  set (asserted by `test_get_user_permissions_no_roles_returns_empty_set`,
  `test_security_gaps.py:112-114`).

## 5. HIPAA Safe Harbor de-identification (HIPAACompliance)

`HIPAACompliance` (`security/__init__.py:191`) implements a **Safe Harbor-style
identifier removal** filter.

- **Identifier list.** `SAFE_HARBOR_FIELDS` (`security/__init__.py:194-213`)
  contains 18 field names: `name`, `address`, `dates`, `telephone`, `fax`,
  `email`, `ssn`, `mrn`, `health_plan`, `account`, `certificate`, `vehicle`,
  `device`, `url`, `ip`, `biometric`, `photo`, `unique_id`.
- **What it does.** `deidentify()` returns a new dict with any key whose
  lowercase form is in `SAFE_HARBOR_FIELDS` removed
  (`security/__init__.py:215-220`). `is_deidentified()` returns `True` only if
  no key matches (`security/__init__.py:222-225`).
- **What it does NOT do.** This is a key-name filter only. It is **not** a
  substitute for the HIPAA **Expert Determination** method, and it does not:
  - handle nested objects or arrays (only top-level dict keys are checked);
  - detect identifiers embedded in free-text fields;
  - address quasi-identifiers (e.g. rare diagnoses, precise ages, postal codes)
    that can re-identify when combined;
  - guarantee that the output is non-reidentifiable.

## 6. Not implemented

The following controls are **absent** from the code reviewed. Their absence is
not a claim that they are unnecessary — it is a statement that this codebase does
not provide them.

- **No TLS termination.** The API is a plain Python facade
  (`api/__init__.py:19`); there is no HTTP server, no TLS, and no certificate
  handling. Any deployment must terminate TLS in front of it.
- **No at-rest database encryption.** Patients are held in an in-memory dict
  (`api/__init__.py:32`); there is no database layer and no encryption of data at
  rest. `EncryptionService` exists but is not wired to any persistence path in the
  reviewed code.
- **No OAuth2 / SMART-on-FHIR authentication.** There is no authentication
  middleware, no token validation, and no SMART-on-FHIR integration. `RBACService`
  checks permissions but does not authenticate the caller; `user_id` is supplied
  by the caller.
- **No rate limiting.** No throttle, quota, or rate-limit logic exists in the
  reviewed code.
- **No secrets manager integration.** The master key is passed as raw `bytes` to
  the `EncryptionService` constructor (`security/__init__.py:25`); there is no
  integration with a secrets manager, HSM, or key-management service.
- **No audit persistence.** The audit trail is in-memory only (see §3); events
  are lost on restart and are not exported.

## 7. Before production — checklist

- [ ] Terminate TLS in front of the API (reverse proxy or gateway).
- [ ] Add authentication (OAuth2 / SMART-on-FHIR) and wire it to `RBACService`.
- [ ] Persist the audit trail to write-once / externally anchored storage.
- [ ] Integrate a secrets manager or HSM for the master key; never pass raw bytes.
- [ ] Add rate limiting and request throttling.
- [ ] Encrypt data at rest; wire `EncryptionService` to the persistence layer.
- [ ] Extend de-identification to nested data and free text; obtain Expert
  Determination where required.
- [ ] Emit audit events from all PHI-touching endpoints, not just
  `acknowledge_alert`.
- [ ] Add logging redaction so PHI does not appear in application logs.
