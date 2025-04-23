import os
import sqlite3
from tkinter import messagebox

def save_results_to_db(ga):
    if not os.path.exists('results'):
        os.makedirs('results')
    
    db_path = os.path.join('results', 'results.db')

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS results (
                    generation INTEGER, 
                    best_value REAL
                 )''')

    results = [(i + 1, ga.best_values[i]) for i in range(len(ga.best_values))]

    c.executemany("INSERT INTO results (generation, best_value) VALUES (?, ?)", results)

    conn.commit()
    conn.close()

    messagebox.showinfo("Info", "Wyniki zapisano do bazy danych")
