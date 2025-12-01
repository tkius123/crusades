# API Guide

The Templar Tournament provides a REST API for querying submission status and leaderboard.

## Running the API Server

### Development

```bash
uv run python -m api.app
```

### With Custom Host/Port

```bash
TOURNAMENT_API_HOST=0.0.0.0 TOURNAMENT_API_PORT=8080 uv run python -m api.app
```

### Production (with uvicorn)

```bash
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
    "status": "ok"
}
```

---

### Get Leaderboard

```
GET /leaderboard?limit=100
```

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `limit` | int | 100 | Maximum entries to return |

**Response:**
```json
[
    {
        "rank": 1,
        "submission_id": "abc123-def456-...",
        "miner_hotkey": "5ABC...XYZ",
        "miner_uid": 42,
        "final_score": 15000.5,
        "num_evaluations": 3,
        "created_at": "2024-01-15T10:30:00Z"
    },
    {
        "rank": 2,
        "submission_id": "xyz789-...",
        "miner_hotkey": "5DEF...ABC",
        "miner_uid": 17,
        "final_score": 14500.2,
        "num_evaluations": 3,
        "created_at": "2024-01-15T09:15:00Z"
    }
]
```

---

### Get Submission Status

```
GET /submissions/{submission_id}
```

**Response:**
```json
{
    "submission_id": "abc123-def456-...",
    "miner_hotkey": "5ABC...XYZ",
    "miner_uid": 42,
    "code_hash": "sha256:abcdef...",
    "status": "finished",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:35:00Z",
    "final_score": 15000.5,
    "error_message": null
}
```

**Status Values:**
| Status | Description |
|--------|-------------|
| `pending` | Awaiting validation |
| `validating` | Being validated |
| `evaluating` | Passed validation, being evaluated |
| `finished` | Evaluation complete, has final score |
| `failed_validation` | Code validation failed |
| `error` | Unexpected error |

---

### Get Submission Evaluations

```
GET /submissions/{submission_id}/evaluations
```

**Response:**
```json
[
    {
        "evaluation_id": "eval-123-...",
        "submission_id": "abc123-def456-...",
        "evaluator_hotkey": "5VAL...123",
        "tokens_per_second": 15100.0,
        "total_tokens": 1510000,
        "wall_time_seconds": 100.0,
        "success": true,
        "error": null,
        "created_at": "2024-01-15T10:32:00Z"
    },
    {
        "evaluation_id": "eval-456-...",
        "submission_id": "abc123-def456-...",
        "evaluator_hotkey": "5VAL...456",
        "tokens_per_second": 14900.0,
        "total_tokens": 1490000,
        "wall_time_seconds": 100.0,
        "success": true,
        "error": null,
        "created_at": "2024-01-15T10:33:00Z"
    }
]
```

## Error Responses

### 404 Not Found

```json
{
    "detail": "Submission not found"
}
```

### 500 Internal Server Error

```json
{
    "detail": "Internal server error"
}
```

## Example Usage

### Python

```python
import httpx

async def get_leaderboard():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/leaderboard")
        return response.json()

async def check_submission(submission_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://localhost:8000/submissions/{submission_id}")
        return response.json()
```

### curl

```bash
# Get leaderboard
curl http://localhost:8000/leaderboard

# Get submission status
curl http://localhost:8000/submissions/abc123-def456

# Get evaluations
curl http://localhost:8000/submissions/abc123-def456/evaluations
```

### JavaScript

```javascript
// Get leaderboard
const response = await fetch('http://localhost:8000/leaderboard');
const leaderboard = await response.json();

// Get submission status
const status = await fetch(`http://localhost:8000/submissions/${submissionId}`)
    .then(r => r.json());
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TOURNAMENT_API_HOST` | `0.0.0.0` | Bind address |
| `TOURNAMENT_API_PORT` | `8000` | Listen port |
| `TOURNAMENT_DEBUG` | `false` | Enable debug mode (auto-reload) |

## OpenAPI Documentation

When the server is running, interactive API docs are available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
