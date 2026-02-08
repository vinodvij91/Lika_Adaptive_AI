#!/usr/bin/env python3
"""
Bulk import materials from CSV into the material_entities table.
Handles 18,330+ materials efficiently with batch inserts.
"""

import csv
import json
import os
import uuid
from datetime import datetime

import psycopg2
from psycopg2.extras import execute_values

# Database connection
DATABASE_URL = os.environ.get("DATABASE_URL")

# Type mapping from CSV types to enum values
TYPE_MAPPING = {
    "Binary_Oxide": "binary_oxide",
    "Binary_Chalcogenide": "binary_chalcogenide",
    "Binary_Pnictide": "binary_pnictide",
    "Thin_Film": "thin_film",
    "Doped_Semiconductor": "doped_semiconductor",
    "Composite": "composite",
    "Homopolymer": "homopolymer",
    "Copolymer": "copolymer",
    "Binary_Alloy": "binary_alloy",
    "Ternary_Alloy": "ternary_alloy",
    "High_Entropy_Alloy": "high_entropy_alloy",
    "Battery_Cathode": "battery_cathode",
    "Battery_Anode": "battery_anode",
    "Solid_Electrolyte": "solid_electrolyte",
    "Perovskite": "perovskite",
    "Double_Perovskite": "double_perovskite",
    "MXene_2D": "mxene_2d",
    "TMD_2D": "tmd_2d",
    "Spinel": "spinel",
    "2D_Material": "2d_material",
}

def parse_float(val):
    """Safely parse float value"""
    if val and val.strip():
        try:
            return float(val)
        except ValueError:
            return None
    return None

def parse_int(val):
    """Safely parse int value"""
    if val and val.strip():
        try:
            return int(val)
        except ValueError:
            return None
    return None

def process_row(row):
    """Process a CSV row into a material entity"""
    csv_type = row.get("type", "").strip()
    material_type = TYPE_MAPPING.get(csv_type, "crystal")  # Default to crystal
    
    # Build representation based on material type
    representation = {}
    if row.get("formula"):
        representation["formula"] = row["formula"]
    if row.get("composition"):
        representation["composition"] = row["composition"]
    if row.get("lattice_a"):
        representation["lattice"] = {
            "a": parse_float(row.get("lattice_a")),
            "b": parse_float(row.get("lattice_b")),
            "c": parse_float(row.get("lattice_c")),
        }
    if row.get("space_group"):
        representation["space_group"] = row["space_group"]
    
    # Build metadata
    metadata = {
        "source": row.get("source", ""),
        "material_id": row.get("material_id", ""),
        "original_type": csv_type,
    }
    
    # Add type-specific metadata
    if row.get("n_elements"):
        metadata["n_elements"] = parse_int(row["n_elements"])
    if row.get("molecular_weight"):
        metadata["molecular_weight"] = parse_float(row["molecular_weight"])
    if row.get("doping_concentration"):
        metadata["doping_concentration"] = row["doping_concentration"]
    if row.get("dopant"):
        metadata["dopant"] = row["dopant"]
    if row.get("doping_level"):
        metadata["doping_level"] = row["doping_level"]
    if row.get("thickness_nm"):
        metadata["thickness_nm"] = parse_float(row["thickness_nm"])
    if row.get("deposition_method"):
        metadata["deposition_method"] = row["deposition_method"]
    if row.get("substrate"):
        metadata["substrate"] = row["substrate"]
    if row.get("layers"):
        metadata["layers"] = parse_int(row["layers"])
    if row.get("temperature"):
        metadata["temperature"] = parse_float(row["temperature"])
    if row.get("matrix"):
        metadata["matrix"] = row["matrix"]
    if row.get("reinforcement"):
        metadata["reinforcement"] = row["reinforcement"]
    if row.get("volume_fraction"):
        metadata["volume_fraction"] = parse_float(row["volume_fraction"])
    if row.get("weight_percent"):
        metadata["weight_percent"] = parse_float(row["weight_percent"])
    if row.get("pdi"):
        metadata["pdi"] = parse_float(row["pdi"])
    if row.get("termination"):
        metadata["termination"] = row["termination"]
    
    # Determine base family
    base_family = csv_type.replace("_", " ")
    
    return {
        "id": str(uuid.uuid4()),
        "name": row.get("name") or row.get("formula") or row.get("material_id"),
        "type": material_type,
        "representation": json.dumps(representation),
        "base_family": base_family,
        "metadata": json.dumps(metadata),
        "is_curated": True,
        "company_id": None,
        "is_demo": False,
    }

def import_materials(csv_path: str, batch_size: int = 500):
    """Import materials from CSV file"""
    print(f"Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    print(f"Reading CSV file: {csv_path}")
    materials = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            materials.append(process_row(row))
    
    print(f"Processed {len(materials)} materials")
    
    # Batch insert
    insert_sql = """
        INSERT INTO material_entities (id, name, type, representation, base_family, metadata, is_curated, company_id, is_demo)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """
    
    total_inserted = 0
    for i in range(0, len(materials), batch_size):
        batch = materials[i:i+batch_size]
        values = [
            (m["id"], m["name"], m["type"], m["representation"], m["base_family"], 
             m["metadata"], m["is_curated"], m["company_id"], m["is_demo"])
            for m in batch
        ]
        execute_values(cur, insert_sql, values)
        conn.commit()
        total_inserted += len(batch)
        print(f"Inserted batch {i//batch_size + 1}: {total_inserted}/{len(materials)} materials")
    
    # Get final count
    cur.execute("SELECT COUNT(*) FROM material_entities")
    count = cur.fetchone()[0]
    
    # Get type breakdown
    cur.execute("""
        SELECT type, COUNT(*) 
        FROM material_entities 
        GROUP BY type 
        ORDER BY COUNT(*) DESC 
        LIMIT 15
    """)
    type_breakdown = cur.fetchall()
    
    cur.close()
    conn.close()
    
    print(f"\n=== Import Complete ===")
    print(f"Total materials in database: {count}")
    print(f"\nMaterial type breakdown:")
    for t, c in type_breakdown:
        print(f"  {t}: {c:,}")
    
    return count

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "attached_assets/materials_library_massive_1769468842456.csv"
    import_materials(csv_path)
