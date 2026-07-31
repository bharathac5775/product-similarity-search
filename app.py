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
from fastapi.responses import HTMLResponse

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
                "image_url": eng.df.loc[pid, "image_url"],
            }
            for pid in ids
        ]
        return {"product_id": product_id, "num_similar": num_similar, "results": results}

    @app.get("/gallery", response_class=HTMLResponse)
    def gallery(
        product_id: str = Query(..., min_length=1),
        num_similar: int = Query(6, gt=0),
    ):
        """A simple visual page: shows the query product and its similar products
        as actual images, so you can *see* the similarity."""
        eng: SimilarityEngine = app.state.engine
        try:
            ids = eng.find_similar_products(product_id, num_similar)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        def card(pid: str, highlight: bool = False) -> str:
            row = eng.df.loc[pid]
            border = "3px solid #2563eb" if highlight else "1px solid #ddd"
            tag = "<div style='color:#2563eb;font-weight:700'>QUERY</div>" if highlight else ""
            return (
                f"<div style='border:{border};border-radius:8px;padding:10px;width:210px;"
                f"box-shadow:0 1px 4px rgba(0,0,0,.08)'>{tag}"
                f"<img src='{row['image_url']}' style='width:100%;height:200px;object-fit:contain;background:#fafafa' "
                f"onerror=\"this.style.opacity=.3;this.alt='(image unavailable)'\"/>"
                f"<div style='font-size:12px;margin-top:6px;line-height:1.3'>{row['product_name'][:70]}</div>"
                f"<div style='font-size:11px;color:#888;margin-top:4px'>brand: {row['brand'] or '—'}</div></div>"
            )

        cards = "".join(card(pid) for pid in ids)
        html = (
            "<html><head><title>Similar products</title></head>"
            "<body style='font-family:system-ui,sans-serif;margin:24px'>"
            "<h2>Query product</h2>"
            f"<div style='display:flex;gap:16px;flex-wrap:wrap'>{card(product_id, highlight=True)}</div>"
            f"<h2 style='margin-top:28px'>Top {len(ids)} similar products</h2>"
            f"<div style='display:flex;gap:16px;flex-wrap:wrap'>{cards}</div>"
            "</body></html>"
        )
        return HTMLResponse(html)


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
