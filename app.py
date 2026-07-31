"""FastAPI microservice wrapping the product similarity engine.

Two entry points share the same route definitions:

* ``build_app(engine)`` — inject a prebuilt engine (used by tests with the small
  fixture, so the 74 MB dataset is never loaded during testing).
* ``app`` — the production application; a lifespan handler builds the engine from
  the configured dataset once at startup and stores it on ``app.state.engine``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from product_similarity.config import settings
from product_similarity.engine import SimilarityEngine


def _register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/find_similar_products")
    def find_similar_products(
        product_id: str = Query(..., min_length=1),
        num_similar: int = Query(..., gt=0),
    ):
        eng: SimilarityEngine = app.state.engine
        try:
            ids = eng.find_similar_products(product_id, num_similar)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"product_id '{product_id}' not found"
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        results = [
            {
                "uniq_id": pid,
                "product_name": eng.df.loc[pid, "product_name"],
                "brand": eng.df.loc[pid, "brand"],
            }
            for pid in ids
        ]
        return {"product_id": product_id, "num_similar": num_similar, "results": results}


def build_app(engine: SimilarityEngine) -> FastAPI:
    """Create an app around an already-built engine (used by tests)."""
    app = FastAPI(title="Product Similarity Search")
    app.state.engine = engine
    _register_routes(app)
    return app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the engine once, at startup, from the configured dataset.
    app.state.engine = SimilarityEngine.from_path(settings.data_path)
    yield


app = FastAPI(title="Product Similarity Search", lifespan=lifespan)
_register_routes(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
