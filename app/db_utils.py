"""
This module provides database utility functions for a PostgreSQL database.

It handles database connections, schema initialization (creating tables for PSEG bills),
and insertion of extracted PSEG bill data.
"""
import psycopg2
import logging
import json 
from decimal import Decimal 
from typing import List, Optional, Any

from app.llm_response_models import PSEGData
from app.config import settings # Import settings

logger = logging.getLogger(__name__)
# DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://pseguser:psegpassword@localhost:5432/psegdb")
# Use DATABASE_URL from settings

# Schema for PSEG bills
CREATE_PSEG_DATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pseg_bills (
    id SERIAL PRIMARY KEY,
    account_number VARCHAR(255),
    customer_name TEXT,
    service_address TEXT,
    billing_address TEXT,
    billing_date DATE,
    billing_period_start_date DATE,
    billing_period_end_date DATE,
    due_date DATE,
    total_amount_due NUMERIC(10, 2),
    previous_balance NUMERIC(10, 2),
    payments_received NUMERIC(10, 2),
    current_charges NUMERIC(10, 2),
    line_items JSONB,
    raw_text_summary TEXT,
    raw_ocr_text TEXT,
    document_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.

    Uses the `DATABASE_URL` from app.config.settings.
    Raises an exception if the connection fails, allowing callers to handle it.
    """
    try:
        # Ensure DATABASE_URL is a string
        db_url = str(settings.database_url) if settings.database_url else None
        if not db_url:
            logger.error("DATABASE_URL is not configured.")
            raise ValueError("DATABASE_URL is not configured.")
        conn = psycopg2.connect(db_url)
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to PostgreSQL database: {e}")
        raise
    except ValueError as e: # Catch misconfigured DATABASE_URL
        logger.error(str(e))
        raise


def initialize_database():
    """
    Initializes the database by creating the PSEG bills table if it doesn't exist.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            logger.info("Initializing PSEG bills table...")
            cur.execute(CREATE_PSEG_DATA_TABLE_SQL)
            conn.commit()
        logger.info("Database initialization complete (PSEG table checked/created).")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        if conn: conn.rollback()
    finally:
        if conn:
            conn.close()

def insert_pseg_data(pseg_data: PSEGData, full_ocr_text: str, doc_type: str) -> Optional[int]:
    """
    Inserts PSEG bill data into the `pseg_bills` table.

    Takes a `PSEGData` object, the full OCR text, and the document type.
    Returns the ID of the newly inserted row, or None if insertion fails.
    """
    conn = None
    sql = """
    INSERT INTO pseg_bills (
        account_number, customer_name, service_address, billing_address, 
        billing_date, billing_period_start_date, billing_period_end_date, 
        due_date, total_amount_due, previous_balance, payments_received, current_charges,
        line_items, raw_text_summary, raw_ocr_text, document_type
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            line_items_json = json.dumps([item.model_dump() for item in pseg_data.line_items]) if pseg_data.line_items else None
            
            cur.execute(sql, (
                pseg_data.account_number,
                pseg_data.customer_name,
                pseg_data.service_address,
                pseg_data.billing_address,
                pseg_data.billing_date,
                pseg_data.billing_period_start_date,
                pseg_data.billing_period_end_date,
                pseg_data.due_date,
                pseg_data.total_amount_due,
                pseg_data.previous_balance,
                pseg_data.payments_received,
                pseg_data.current_charges,
                line_items_json,
                pseg_data.raw_text_summary,
                full_ocr_text,
                doc_type
            ))
            inserted_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Inserted PSEG data with ID: {inserted_id}")
            return inserted_id
    except Exception as e:
        logger.error(f"Error inserting PSEG data: {e}", exc_info=True)
        if conn: conn.rollback()
        return None
    finally:
        if conn: conn.close()

