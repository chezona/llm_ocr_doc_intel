"""
This module provides database utility functions for a PostgreSQL database.

It handles database connections, schema initialization (creating tables for PSEG bills),
and insertion of extracted PSEG bill data.
"""
import psycopg2
import os
import logging
import json # For metadata JSONB field (retained if PSEG table needs it, currently not)
from decimal import Decimal # For currency fields (retained for PSEGData)
from typing import List, Optional

from app.models import PSEGData
# Receipt model imports removed

logger = logging.getLogger(__name__)
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://pseguser:psegpassword@localhost:5432/psegdb")

# Schema for PSEG bills
CREATE_PSEG_DATA_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pseg_bills (
    id SERIAL PRIMARY KEY,
    account_number VARCHAR(255),
    bill_date VARCHAR(50),
    due_date VARCHAR(50),
    service_address TEXT,
    total_amount_due NUMERIC(10, 2),
    previous_balance NUMERIC(10, 2),
    payments_received NUMERIC(10, 2),
    current_charges NUMERIC(10, 2),
    raw_ocr_text TEXT,
    document_type VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

# FULL_DB_SCHEMA_SQL for receipts, merchants, items, etc. has been removed.

def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.

    Uses the `DATABASE_URL` environment variable for connection parameters.
    Raises an exception if the connection fails, allowing callers to handle it.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Could not connect to PostgreSQL database: {e}")
        raise

def initialize_database():
    """
    Initializes the database by creating the PSEG bills table if it doesn't exist.

    Connects to the database and executes CREATE TABLE IF NOT EXISTS DDL statement
    for PSEG bills.
    Logs success or errors encountered during the initialization process.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            logger.info("Initializing PSEG bills table...")
            cur.execute(CREATE_PSEG_DATA_TABLE_SQL)
            # Removed execution of FULL_DB_SCHEMA_SQL
            conn.commit()
        logger.info("Database initialization complete (PSEG table checked/created).")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        if conn: conn.rollback()
    finally:
        if conn:
            conn.close()

def insert_pseg_data(pseg_data: PSEGData, raw_text: str, doc_type: str) -> Optional[int]:
    """
    Inserts PSEG bill data into the `pseg_bills` table.

    Takes a `PSEGData` object, the raw OCR text, and the document type.
    Returns the ID of the newly inserted row, or None if insertion fails.
    """
    conn = None
    sql = """
    INSERT INTO pseg_bills (
        account_number, bill_date, due_date, service_address, 
        total_amount_due, previous_balance, payments_received, current_charges, 
        raw_ocr_text, document_type
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(sql, (
                pseg_data.account_number, pseg_data.bill_date, pseg_data.due_date,
                pseg_data.service_address, pseg_data.total_amount_due,
                pseg_data.previous_balance, pseg_data.payments_received,
                pseg_data.current_charges, raw_text, doc_type
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

if __name__ == '__main__':
    """
    Main execution block for direct script testing.

    When this script is run directly, it attempts to:
    1. Initialize the database (i.e., create the 'pseg_bills' table if it doesn't exist).
    2. Optionally, run example PSEG data insertion (currently commented out).
    """
    logger.info("Attempting to initialize database (PSEG table)...")
    initialize_database()
    logger.info("Database initialization attempt complete.")

    # Example of inserting PSEG data (uncomment and modify to test)
    # logger.info("\nAttempting to insert sample PSEG data...")
    # sample_pseg_data = PSEGData(
    #     account_number="1234567890",
    #     bill_date="2024-07-27",
    #     due_date="2024-08-15",
    #     service_address="100 Main St, Anytown, NJ",
    #     total_amount_due=Decimal("150.75"),
    #     previous_balance=Decimal("20.00"),
    #     payments_received=Decimal("20.00"),
    #     current_charges=Decimal("150.75")
    # )
    # inserted_pseg_id = insert_pseg_data(
    #     pseg_data=sample_pseg_data, 
    #     raw_text="Sample raw OCR text for PSEG bill...", 
    #     doc_type="pseg_bill"
    # )
    # if inserted_pseg_id:
    #     print(f"Successfully inserted sample PSEG data with ID: {inserted_pseg_id}")
    # else:
    #     print("Failed to insert sample PSEG data.") 