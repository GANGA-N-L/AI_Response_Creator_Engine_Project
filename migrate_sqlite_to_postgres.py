import sqlite3

from sqlalchemy import text

from app.database.database import SessionLocal
from app.models.customer import Customer
from app.models.prompt import Prompt


SQLITE_DATABASE = "ai_response_creator.db"


def migrate():
    sqlite_connection = sqlite3.connect(SQLITE_DATABASE)
    sqlite_connection.row_factory = sqlite3.Row

    db = SessionLocal()

    try:
        # ---------------------------------------------------------
        # Read customers and prompts from SQLite
        # ---------------------------------------------------------
        customers = sqlite_connection.execute(
            """
            SELECT id, customer_name, credits
            FROM customers
            ORDER BY id
            """
        ).fetchall()

        prompts = sqlite_connection.execute(
            """
            SELECT id, prompt_name, prompt_text
            FROM prompts
            ORDER BY id
            """
        ).fetchall()

        print(f"SQLite customers found: {len(customers)}")
        print(f"SQLite prompts found: {len(prompts)}")

        # ---------------------------------------------------------
        # Make sure PostgreSQL target tables are empty
        # ---------------------------------------------------------
        existing_customers = db.query(Customer).count()
        existing_prompts = db.query(Prompt).count()

        if existing_customers != 0 or existing_prompts != 0:
            raise RuntimeError(
                "PostgreSQL target tables are not empty. "
                "Migration aborted to prevent duplicate data."
            )

        # ---------------------------------------------------------
        # Insert customers
        # ---------------------------------------------------------
        for customer in customers:
            db.add(
                Customer(
                    id=customer["id"],
                    customer_name=customer["customer_name"],
                    credits=customer["credits"]
                )
            )

        # ---------------------------------------------------------
        # Insert prompts
        # ---------------------------------------------------------
        for prompt in prompts:
            db.add(
                Prompt(
                    id=prompt["id"],
                    prompt_name=prompt["prompt_name"],
                    prompt_text=prompt["prompt_text"]
                )
            )

        # ---------------------------------------------------------
        # Commit data
        # ---------------------------------------------------------
        db.commit()

        # ---------------------------------------------------------
        # Reset PostgreSQL sequences
        # ---------------------------------------------------------
        db.execute(
            text(
                """
                SELECT setval(
                    pg_get_serial_sequence('customers', 'id'),
                    COALESCE((SELECT MAX(id) FROM customers), 1),
                    true
                )
                """
            )
        )

        db.execute(
            text(
                """
                SELECT setval(
                    pg_get_serial_sequence('prompts', 'id'),
                    COALESCE((SELECT MAX(id) FROM prompts), 1),
                    true
                )
                """
            )
        )

        db.commit()

        print("Data migration completed successfully.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()
        sqlite_connection.close()


if __name__ == "__main__":
    migrate()