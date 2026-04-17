"""
analytics_service.py
Tracks search queries, performance metrics, and provides analytics for the K-12
image database. Helps identify popular topics, optimize indexing, and debug issues.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class SearchAnalytics:
    """Collects and analyzes search query statistics."""

    def __init__(self, log_file: str = "search_analytics.jsonl"):
        """Initialize analytics logger.
        
        Args:
            log_file: Path to JSONL file for storing query logs
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_stats = defaultdict(int)
        logger.info(f"Analytics logger initialized: {self.log_file}")

    def log_query(
        self,
        query_text: str,
        num_results: int,
        response_time_ms: float,
        filters: Optional[Dict[str, Any]] = None,
        results_quality: Optional[float] = None,
    ) -> None:
        """Log a search query event.
        
        Args:
            query_text: The search query
            num_results: Number of results returned
            response_time_ms: Query execution time in milliseconds
            filters: Applied filters (subject, grade, etc.)
            results_quality: Optional score indicating result quality (0-1)
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query_text[:200],  # Truncate for privacy
            "num_results": num_results,
            "response_time_ms": round(response_time_ms, 2),
            "filters": filters or {},
            "quality_score": results_quality,
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
            self.session_stats["queries_logged"] += 1
        except Exception as e:
            logger.error(f"Failed to log query: {e}")

    def log_error(
        self,
        query_text: str,
        error_message: str,
        error_type: str,
    ) -> None:
        """Log a search error.
        
        Args:
            query_text: The problematic query
            error_message: Error details
            error_type: Category of error (e.g., 'timeout', 'validation', 'database')
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error",
            "query": query_text[:200],
            "error_type": error_type,
            "error_message": str(error_message)[:500],
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
            self.session_stats["errors_logged"] += 1
        except Exception as e:
            logger.error(f"Failed to log error: {e}")

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Generate a summary of analytics from the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with statistics including top queries, error rates, etc.
        """
        if not self.log_file.exists():
            return {"error": "No analytics data available"}

        cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - days)

        queries = []
        errors = []
        response_times = []

        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        event_dt = datetime.fromisoformat(event.get("timestamp", ""))
                        if event_dt < cutoff:
                            continue

                        if event.get("type") == "error":
                            errors.append(event)
                        elif "query" in event:
                            queries.append(event)
                            if event.get("response_time_ms"):
                                response_times.append(event["response_time_ms"])
                    except json.JSONDecodeError:
                        continue

            # Compute statistics
            query_texts = [q["query"] for q in queries]
            top_queries = Counter(query_texts).most_common(10)
            error_types = Counter(e.get("error_type", "unknown") for e in errors)

            avg_response_time = (
                sum(response_times) / len(response_times)
                if response_times
                else 0
            )

            return {
                "period_days": days,
                "total_queries": len(queries),
                "total_errors": len(errors),
                "error_rate": len(errors) / (len(queries) + len(errors))
                if (len(queries) + len(errors)) > 0
                else 0,
                "avg_response_time_ms": round(avg_response_time, 2),
                "top_queries": [{"query": q, "count": c} for q, c in top_queries],
                "error_breakdown": dict(error_types),
            }

        except Exception as e:
            logger.error(f"Failed to compute summary: {e}")
            return {"error": str(e)}

    def export_report(self, output_file: str = "analytics_report.json") -> None:
        """Export a detailed analytics report.
        
        Args:
            output_file: Path to write the JSON report
        """
        summary = self.get_summary(days=30)
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "analytics": summary,
            "session_stats": dict(self.session_stats),
        }

        try:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Analytics report exported to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
