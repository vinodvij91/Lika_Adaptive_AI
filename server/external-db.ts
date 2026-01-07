import pg from "pg";

const { Pool } = pg;

const externalDbUrl = process.env.EXTERNAL_DATABASE_URL;

if (!externalDbUrl) {
  console.warn("EXTERNAL_DATABASE_URL not set - external auth will not work");
}

export const externalPool = externalDbUrl 
  ? new Pool({ 
      connectionString: externalDbUrl,
      ssl: { rejectUnauthorized: false }
    }) 
  : null;

export interface ExternalUser {
  id: string;
  username: string;
  password: string;
  company_id: string;
  role: string;
}

export interface ExternalCompany {
  id: string;
  name: string;
  logo: string | null;
  gpu_allocated: number;
  cpu_allocated: number;
  active_jobs: number;
}

export async function findUserByUsername(username: string): Promise<ExternalUser | null> {
  if (!externalPool) return null;
  
  const result = await externalPool.query<ExternalUser>(
    'SELECT id, username, password, company_id, role FROM public.users WHERE username = $1',
    [username]
  );
  
  return result.rows[0] || null;
}

export async function findCompanyById(companyId: string): Promise<ExternalCompany | null> {
  if (!externalPool) return null;
  
  const result = await externalPool.query<ExternalCompany>(
    'SELECT id, name, logo, gpu_allocated, cpu_allocated, active_jobs FROM public.companies WHERE id = $1',
    [companyId]
  );
  
  return result.rows[0] || null;
}

export async function getSmilesCountForCompany(companyId: string): Promise<number> {
  if (!externalPool) return 0;
  
  const result = await externalPool.query(
    'SELECT COUNT(*) as count FROM public.smiles_library WHERE company_id = $1',
    [companyId]
  );
  
  return parseInt(result.rows[0]?.count || '0', 10);
}

export async function getActiveJobsForCompany(companyId: string): Promise<number> {
  if (!externalPool) return 0;
  
  const result = await externalPool.query(
    "SELECT COUNT(*) as count FROM public.processing_jobs WHERE company_id = $1 AND status != 'completed'",
    [companyId]
  );
  
  return parseInt(result.rows[0]?.count || '0', 10);
}

export async function getSshConfigsCountForCompany(companyId: string): Promise<number> {
  if (!externalPool) return 0;
  
  const result = await externalPool.query(
    'SELECT COUNT(*) as count FROM public.ssh_configs WHERE company_id = $1',
    [companyId]
  );
  
  return parseInt(result.rows[0]?.count || '0', 10);
}

export async function getDiseasesCount(): Promise<number> {
  if (!externalPool) return 0;
  
  const result = await externalPool.query(
    'SELECT COUNT(DISTINCT id) as count FROM public.diseases'
  );
  
  return parseInt(result.rows[0]?.count || '0', 10);
}
