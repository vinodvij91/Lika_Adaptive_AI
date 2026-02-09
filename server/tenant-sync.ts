import { externalPool } from "./external-db";

export interface TenantInfo {
  tenantId: string;
  role: string;
}

export async function syncAuth0UserAndResolveTenant(
  sub: string,
  email: string,
  name: string
): Promise<TenantInfo> {
  if (!externalPool) {
    console.warn("External DB not connected — defaulting tenant to jnj");
    return { tenantId: "jnj", role: "scientist" };
  }

  try {
    const domain = email.split("@")[1] || "";

    const tenantResult = await externalPool.query(
      "SELECT id FROM tenants WHERE primary_domain = $1",
      [domain]
    );

    let tenantId: string;
    if (tenantResult.rows.length > 0) {
      tenantId = tenantResult.rows[0].id;
    } else {
      tenantId = "jnj";
      console.warn(`No tenant found for domain ${domain}, defaulting to jnj`);
    }

    await externalPool.query(
      `INSERT INTO auth0_users (id, email, name)
       VALUES ($1, $2, $3)
       ON CONFLICT (id) DO UPDATE SET email = $2, name = $3`,
      [sub, email, name || ""]
    );

    const tuResult = await externalPool.query(
      `INSERT INTO tenant_users (tenant_id, user_id, role)
       VALUES ($1, $2, 'scientist')
       ON CONFLICT (tenant_id, user_id) DO NOTHING
       RETURNING role`,
      [tenantId, sub]
    );

    let role = "scientist";
    if (tuResult.rows.length === 0) {
      const existing = await externalPool.query(
        "SELECT role FROM tenant_users WHERE tenant_id = $1 AND user_id = $2",
        [tenantId, sub]
      );
      if (existing.rows.length > 0) {
        role = existing.rows[0].role;
      }
    }

    return { tenantId, role };
  } catch (error) {
    console.error("Tenant sync error:", error);
    return { tenantId: "jnj", role: "scientist" };
  }
}

export async function lookupTenantForUser(userId: string): Promise<TenantInfo> {
  if (!externalPool) {
    return { tenantId: "jnj", role: "scientist" };
  }

  try {
    const result = await externalPool.query(
      "SELECT tenant_id, role FROM tenant_users WHERE user_id = $1 LIMIT 1",
      [userId]
    );

    if (result.rows.length > 0) {
      return {
        tenantId: result.rows[0].tenant_id,
        role: result.rows[0].role,
      };
    }
  } catch (error) {
    console.error("Tenant lookup error:", error);
  }

  return { tenantId: "jnj", role: "scientist" };
}
