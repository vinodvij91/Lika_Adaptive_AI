import { createClient, SupabaseClient } from '@supabase/supabase-js';
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
  private supabaseClient: SupabaseClient | null = null;
  private digitalOceanPool: pg.Pool | null = null;

  private getSupabaseClient(): SupabaseClient {
    if (!this.supabaseClient) {
      const supabaseUrl = process.env.SUPABASE_URL;
      const supabaseKey = process.env.SUPABASE_ANON_KEY;
      
      if (!supabaseUrl || !supabaseKey) {
        throw new Error('SUPABASE_URL and SUPABASE_ANON_KEY environment variables are required');
      }
      
      this.supabaseClient = createClient(supabaseUrl, supabaseKey);
    }
    return this.supabaseClient;
  }

  private getDigitalOceanPool(): pg.Pool {
    if (!this.digitalOceanPool) {
      const doUrl = process.env.DIGITALOCEAN_DATABASE_URL;
      if (!doUrl) {
        throw new Error('DIGITALOCEAN_DATABASE_URL environment variable is not set');
      }
      this.digitalOceanPool = new Pool({
        connectionString: doUrl,
        ssl: { rejectUnauthorized: false }
      });
    }
    return this.digitalOceanPool;
  }

  async testConnection(source: 'supabase' | 'digitalocean' = 'supabase'): Promise<{ success: boolean; message: string; tables?: string[]; source: string }> {
    try {
      if (source === 'supabase') {
        const client = this.getSupabaseClient();
        
        const { data, error } = await client.from('Materials Property Table').select('*').limit(1);
        
        if (error) {
          return {
            success: false,
            message: `Connection failed: ${error.message}`,
            source
          };
        }
        
        return {
          success: true,
          message: 'Successfully connected to Supabase',
          tables: ['Materials Property Table'],
          source
        };
      } else {
        const pool = this.getDigitalOceanPool();
        const poolClient = await pool.connect();
        
        const result = await poolClient.query(`
          SELECT table_name 
          FROM information_schema.tables 
          WHERE table_schema = 'public'
          ORDER BY table_name
        `);
        
        poolClient.release();
        
        const tables = result.rows.map(r => r.table_name);
        return {
          success: true,
          message: 'Successfully connected to DigitalOcean database',
          tables,
          source
        };
      }
    } catch (error: any) {
      return {
        success: false,
        message: `Connection failed: ${error.message}`,
        source
      };
    }
  }

  async testAllConnections(): Promise<{ supabase: any; digitalocean: any }> {
    const [supabase, digitalocean] = await Promise.all([
      this.testConnection('supabase').catch(e => ({ success: false, message: e.message, source: 'supabase' })),
      this.testConnection('digitalocean').catch(e => ({ success: false, message: e.message, source: 'digitalocean' }))
    ]);
    
    return { supabase, digitalocean };
  }

  async previewTable(tableName: string, limit: number = 10, source: 'supabase' | 'digitalocean' = 'supabase'): Promise<{ columns: string[]; rows: any[]; totalCount: number }> {
    if (source === 'supabase') {
      const client = this.getSupabaseClient();
      
      const { data, error, count } = await client
        .from(tableName)
        .select('*', { count: 'exact' })
        .limit(limit);
      
      if (error) {
        throw new Error(`Failed to preview table: ${error.message}`);
      }
      
      const columns = data && data.length > 0 ? Object.keys(data[0]) : [];
      
      return {
        columns,
        rows: data || [],
        totalCount: count || 0
      };
    } else {
      const pool = this.getDigitalOceanPool();
      const poolClient = await pool.connect();
      
      try {
        const countResult = await poolClient.query(`SELECT COUNT(*) FROM "${tableName}"`);
        const totalCount = parseInt(countResult.rows[0].count);
        
        const result = await poolClient.query(`SELECT * FROM "${tableName}" LIMIT $1`, [limit]);
        
        const columns = result.fields.map(f => f.name);
        
        return {
          columns,
          rows: result.rows,
          totalCount
        };
      } finally {
        poolClient.release();
      }
    }
  }

  async syncSmiles(tableName: string = 'SMILES'): Promise<SyncResult> {
    const result: SyncResult = {
      success: true,
      table: tableName,
      recordsProcessed: 0,
      recordsInserted: 0,
      recordsSkipped: 0,
      errors: []
    };

    try {
      const client = this.getSupabaseClient();
      
      const { data: rows, error } = await client
        .from(tableName)
        .select('*');
      
      if (error) {
        throw new Error(`Failed to fetch SMILES data: ${error.message}`);
      }

      if (!rows || rows.length === 0) {
        return result;
      }

      for (const row of rows as ExternalSmilesRow[]) {
        result.recordsProcessed++;
        
        try {
          if (!row.SMILES) {
            result.recordsSkipped++;
            continue;
          }

          const existing = await db
            .select()
            .from(molecules)
            .where(eq(molecules.smiles, row.SMILES))
            .limit(1);

          if (existing.length > 0) {
            result.recordsSkipped++;
            continue;
          }

          await db.insert(molecules).values({
            name: row.Drug_Name || 'Unknown',
            smiles: row.SMILES,
            seriesId: row.ChEMBL_ID || null,
            scaffoldId: row.Therapeutic_Class || null,
            source: 'uploaded'
          });

          result.recordsInserted++;
        } catch (rowError: any) {
          result.errors.push(`Row error: ${rowError.message}`);
        }
      }
    } catch (error: any) {
      result.success = false;
      result.errors.push(error.message);
    }

    return result;
  }

  async syncMaterialProperties(tableName: string = 'Materials Property Table', batchSize: number = 1000, maxRecords: number = 600000): Promise<SyncResult> {
    const result: SyncResult = {
      success: true,
      table: tableName,
      recordsProcessed: 0,
      recordsInserted: 0,
      recordsSkipped: 0,
      errors: []
    };

    try {
      const client = this.getSupabaseClient();
      
      // Pre-load all existing materials for fast lookup
      const existingMaterials = await db.select({ id: materialEntities.id, name: materialEntities.name }).from(materialEntities);
      const materialMap = new Map<string, string>();
      for (const mat of existingMaterials) {
        if (mat.name) materialMap.set(mat.name, mat.id);
      }
      
      let offset = 0;
      let hasMore = true;

      while (hasMore && result.recordsProcessed < maxRecords) {
        const { data: rows, error } = await client
          .from(tableName)
          .select('*')
          .range(offset, offset + batchSize - 1);
        
        if (error) {
          throw new Error(`Failed to fetch material properties: ${error.message}`);
        }

        if (!rows || rows.length === 0) {
          hasMore = false;
          break;
        }

        // Collect new materials to insert in batch
        const newMaterials: { name: string; type: 'composite'; representation: any; baseFamily: string; isCurated: boolean; isDemo: boolean }[] = [];
        const seenNewMaterials = new Set<string>();
        
        for (const row of rows as ExternalMaterialPropertiesRow[]) {
          if (!materialMap.has(row.material_id) && !seenNewMaterials.has(row.material_id)) {
            seenNewMaterials.add(row.material_id);
            newMaterials.push({
              name: row.material_id,
              type: 'composite' as const,
              representation: { category: row.category },
              baseFamily: row.category,
              isCurated: true,
              isDemo: false
            });
          }
        }
        
        // Batch insert new materials
        if (newMaterials.length > 0) {
          const inserted = await db.insert(materialEntities).values(newMaterials).returning({ id: materialEntities.id, name: materialEntities.name });
          for (const mat of inserted) {
            if (mat.name) materialMap.set(mat.name, mat.id);
          }
        }

        // Batch insert properties
        const propertiesToInsert = [];
        for (const row of rows as ExternalMaterialPropertiesRow[]) {
          result.recordsProcessed++;
          const materialId = materialMap.get(row.material_id);
          if (materialId) {
            propertiesToInsert.push({
              materialId,
              propertyName: row.property,
              value: parseFloat(String(row.value)) || 0,
              units: row.units,
              source: 'experiment' as const,
              confidence: 1.0
            });
          }
        }
        
        if (propertiesToInsert.length > 0) {
          await db.insert(materialProperties).values(propertiesToInsert);
          result.recordsInserted += propertiesToInsert.length;
        }

        offset += batchSize;
        if (rows.length < batchSize) {
          hasMore = false;
        }
      }
    } catch (error: any) {
      result.success = false;
      result.errors.push(error.message);
    }

    return result;
  }

  async syncVariants(tableName: string = 'Variants_Formulations', source: 'supabase' | 'digitalocean' = 'digitalocean'): Promise<SyncResult> {
    const result: SyncResult = {
      success: true,
      table: tableName,
      recordsProcessed: 0,
      recordsInserted: 0,
      recordsSkipped: 0,
      errors: []
    };

    try {
      let rows: ExternalVariantsRow[] = [];
      
      if (source === 'supabase') {
        const client = this.getSupabaseClient();
        const { data, error } = await client.from(tableName).select('*');
        if (error) throw new Error(error.message);
        rows = data || [];
      } else {
        const pool = this.getDigitalOceanPool();
        const poolClient = await pool.connect();
        try {
          const queryResult = await poolClient.query(`SELECT * FROM "${tableName}"`);
          rows = queryResult.rows;
        } finally {
          poolClient.release();
        }
      }

      if (rows.length === 0) {
        return result;
      }

      const materialMap = new Map<string, string>();

      for (const row of rows) {
        result.recordsProcessed++;
        
        try {
          const baseMaterialName = row.base_material || row.active_material || 'Unknown Material';
          let internalMaterialId = materialMap.get(baseMaterialName);
          
          if (!internalMaterialId) {
            const existing = await db
              .select()
              .from(materialEntities)
              .where(eq(materialEntities.name, baseMaterialName))
              .limit(1);

            if (existing.length > 0) {
              internalMaterialId = existing[0].id;
            } else {
              const inserted = await db.insert(materialEntities).values({
                name: baseMaterialName,
                type: 'composite',
                representation: { variant_source: source },
                baseFamily: row.variant_type || 'formulation',
                isCurated: true,
                isDemo: false
              }).returning({ id: materialEntities.id });
              
              internalMaterialId = inserted[0].id;
            }
            materialMap.set(baseMaterialName, internalMaterialId);
          }

          const { variant_id, variant_type, base_material, active_material, description, process, processing_method, ...restParams } = row;

          await db.insert(materialVariants).values({
            materialId: internalMaterialId!,
            variantParams: {
              external_variant_id: variant_id,
              variant_type: variant_type || 'formulation',
              base_material,
              active_material,
              description,
              process,
              processing_method,
              source: source,
              ...restParams
            },
            generatedBy: 'human'
          });

          result.recordsInserted++;
        } catch (rowError: any) {
          result.errors.push(`Row error: ${rowError.message}`);
          result.recordsSkipped++;
        }
      }
    } catch (error: any) {
      result.success = false;
      result.errors.push(error.message);
    }

    return result;
  }

  async syncAll(): Promise<{ smiles: SyncResult; materialProperties: SyncResult; variants: SyncResult }> {
    const [smiles, materialProperties, variants] = await Promise.all([
      this.syncSmiles().catch(e => ({
        success: false,
        table: 'SMILES',
        recordsProcessed: 0,
        recordsInserted: 0,
        recordsSkipped: 0,
        errors: [e.message]
      })),
      this.syncMaterialProperties().catch(e => ({
        success: false,
        table: 'Materials Property Table',
        recordsProcessed: 0,
        recordsInserted: 0,
        recordsSkipped: 0,
        errors: [e.message]
      })),
      this.syncVariants().catch(e => ({
        success: false,
        table: 'Variants_Formulations',
        recordsProcessed: 0,
        recordsInserted: 0,
        recordsSkipped: 0,
        errors: [e.message]
      }))
    ]);

    return { smiles, materialProperties, variants };
  }
}

export const supabaseSyncService = new SupabaseSyncService();
