import pg from 'pg';
import { db } from '../db';
import { molecules, materialEntities, materialVariants, materialProperties } from '@shared/schema';
import { eq } from 'drizzle-orm';

const { Pool } = pg;

export interface SyncResult {
  success: boolean;
  table: string;
  recordsProcessed: number;
  recordsInserted: number;
  recordsSkipped: number;
  errors: string[];
}

export interface ExternalSmilesRow {
  Drug_Name: string;
  Disease_Condition: string;
  SMILES: string;
  ChEMBL_ID: string;
  Category: string;
  Therapeutic_Class: string;
}

export interface ExternalMaterialPropertiesRow {
  property_id: string;
  material_id: string;
  category: string;
  property: string;
  value: number;
  units: string;
  temp_C: number;
}

export interface ExternalVariantsRow {
  variant_id: string;
  variant_type: string;
  base_material?: string;
  active_material?: string;
  description?: string;
  process?: string;
  processing_method?: string;
  [key: string]: any;
}

class SupabaseSyncService {
  private externalPool: pg.Pool | null = null;

  private getExternalPool(): pg.Pool {
    if (!this.externalPool) {
      const externalUrl = process.env.EXTERNAL_DATABASE_URL;
      if (!externalUrl) {
        throw new Error('EXTERNAL_DATABASE_URL environment variable is not set');
      }
      this.externalPool = new Pool({
        connectionString: externalUrl,
        ssl: { rejectUnauthorized: false }
      });
    }
    return this.externalPool;
  }

  async testConnection(): Promise<{ success: boolean; message: string; tables?: string[] }> {
    try {
      const pool = this.getExternalPool();
      const client = await pool.connect();
      
      const result = await client.query(`
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
      `);
      
      client.release();
      
      const tables = result.rows.map(r => r.table_name);
      return {
        success: true,
        message: 'Successfully connected to external database',
        tables
      };
    } catch (error: any) {
      return {
        success: false,
        message: `Connection failed: ${error.message}`
      };
    }
  }

  async syncSmilesTable(tableName: string = 'SMILES'): Promise<SyncResult> {
    const result: SyncResult = {
      success: false,
      table: tableName,
      recordsProcessed: 0,
      recordsInserted: 0,
      recordsSkipped: 0,
      errors: []
    };

    try {
      const pool = this.getExternalPool();
      const client = await pool.connect();
      
      const queryResult = await client.query<ExternalSmilesRow>(`
        SELECT "Drug_Name", "Disease_Condition", "SMILES", "ChEMBL_ID", "Category", "Therapeutic_Class"
        FROM "${tableName}"
      `);
      client.release();

      for (const row of queryResult.rows) {
        result.recordsProcessed++;
        
        try {
          if (!row.SMILES) {
            result.recordsSkipped++;
            continue;
          }

          const existing = await db.select()
            .from(molecules)
            .where(eq(molecules.smiles, row.SMILES))
            .limit(1);

          if (existing.length > 0) {
            result.recordsSkipped++;
            continue;
          }

          await db.insert(molecules).values({
            smiles: row.SMILES,
            name: row.Drug_Name || null,
            seriesId: row.ChEMBL_ID || null,
            scaffoldId: row.Therapeutic_Class || null,
            source: 'uploaded'
          });
          
          result.recordsInserted++;
        } catch (err: any) {
          result.errors.push(`Row ${result.recordsProcessed}: ${err.message}`);
        }
      }

      result.success = true;
    } catch (error: any) {
      result.errors.push(error.message);
    }

    return result;
  }

  async syncMaterialPropertiesTable(tableName: string = 'Material_Properties'): Promise<SyncResult> {
    const result: SyncResult = {
      success: false,
      table: tableName,
      recordsProcessed: 0,
      recordsInserted: 0,
      recordsSkipped: 0,
      errors: []
    };

    try {
      const pool = this.getExternalPool();
      const client = await pool.connect();
      
      const queryResult = await client.query<ExternalMaterialPropertiesRow>(`
        SELECT property_id, material_id, category, property, value, units, temp_C
        FROM "${tableName}"
      `);
      client.release();

      const materialMap = new Map<string, string>();

      for (const row of queryResult.rows) {
        result.recordsProcessed++;
        
        try {
          let internalMaterialId = materialMap.get(row.material_id);
          
          if (!internalMaterialId) {
            const existing = await db.select()
              .from(materialEntities)
              .where(eq(materialEntities.name, row.material_id))
              .limit(1);

            if (existing.length > 0) {
              internalMaterialId = existing[0].id;
            } else {
              const inserted = await db.insert(materialEntities).values({
                name: row.material_id,
                type: 'other',
                representation: { category: row.category },
                baseFamily: row.category,
                isCurated: true,
                isDemo: false
              }).returning({ id: materialEntities.id });
              
              internalMaterialId = inserted[0].id;
            }
            materialMap.set(row.material_id, internalMaterialId);
          }

          await db.insert(materialProperties).values({
            materialId: internalMaterialId,
            propertyName: row.property,
            value: row.value,
            units: row.units,
            source: 'experiment',
            confidence: 1.0
          });
          
          result.recordsInserted++;
        } catch (err: any) {
          result.errors.push(`Row ${result.recordsProcessed}: ${err.message}`);
        }
      }

      result.success = true;
    } catch (error: any) {
      result.errors.push(error.message);
    }

    return result;
  }

  async syncVariantsFormulationsTable(tableName: string = 'Variants_Formulations'): Promise<SyncResult> {
    const result: SyncResult = {
      success: false,
      table: tableName,
      recordsProcessed: 0,
      recordsInserted: 0,
      recordsSkipped: 0,
      errors: []
    };

    try {
      const pool = this.getExternalPool();
      const client = await pool.connect();
      
      const queryResult = await client.query<ExternalVariantsRow>(`SELECT * FROM "${tableName}"`);
      client.release();

      const materialMap = new Map<string, string>();

      for (const row of queryResult.rows) {
        result.recordsProcessed++;
        
        try {
          const baseMaterialName = row.base_material || row.active_material || `Material_${row.variant_id}`;
          let internalMaterialId = materialMap.get(baseMaterialName);
          
          if (!internalMaterialId) {
            const existing = await db.select()
              .from(materialEntities)
              .where(eq(materialEntities.name, baseMaterialName))
              .limit(1);

            if (existing.length > 0) {
              internalMaterialId = existing[0].id;
            } else {
              const inserted = await db.insert(materialEntities).values({
                name: baseMaterialName,
                type: 'composite',
                representation: { formula: baseMaterialName },
                baseFamily: row.variant_type || 'Unknown',
                isCurated: true,
                isDemo: false
              }).returning({ id: materialEntities.id });
              
              internalMaterialId = inserted[0].id;
            }
            materialMap.set(baseMaterialName, internalMaterialId);
          }

          const { variant_id, variant_type, base_material, active_material, description, ...variantParams } = row;
          
          await db.insert(materialVariants).values({
            materialId: internalMaterialId,
            variantParams: {
              externalId: variant_id,
              variantType: variant_type,
              description: description,
              ...variantParams
            },
            generatedBy: 'human',
            simulationState: 'pending'
          });
          
          result.recordsInserted++;
        } catch (err: any) {
          result.errors.push(`Row ${result.recordsProcessed}: ${err.message}`);
        }
      }

      result.success = true;
    } catch (error: any) {
      result.errors.push(error.message);
    }

    return result;
  }

  async syncAll(): Promise<{ smiles: SyncResult; materialProperties: SyncResult; variants: SyncResult }> {
    const smiles = await this.syncSmilesTable();
    const materialPropertiesResult = await this.syncMaterialPropertiesTable();
    const variants = await this.syncVariantsFormulationsTable();

    return {
      smiles,
      materialProperties: materialPropertiesResult,
      variants
    };
  }

  async getTablePreview(tableName: string, limit: number = 10): Promise<{ success: boolean; data?: any[]; columns?: string[]; error?: string }> {
    try {
      const pool = this.getExternalPool();
      const client = await pool.connect();
      
      const queryResult = await client.query(`SELECT * FROM "${tableName}" LIMIT $1`, [limit]);
      client.release();
      
      return {
        success: true,
        data: queryResult.rows,
        columns: queryResult.fields.map(f => f.name)
      };
    } catch (error: any) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async close(): Promise<void> {
    if (this.externalPool) {
      await this.externalPool.end();
      this.externalPool = null;
    }
  }
}

export const supabaseSyncService = new SupabaseSyncService();
