import { Router, Request, Response } from "express";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import { 
  findUserByUsername, 
  findCompanyById,
  getSmilesCountForCompany,
  getActiveJobsForCompany,
  getSshConfigsCountForCompany,
  getDiseasesCount,
  ExternalUser,
  ExternalCompany
} from "./external-db";

const router = Router();

const JWT_SECRET = process.env.SESSION_SECRET || "lika-sciences-secret-key";
const COOKIE_NAME = "lika_auth_token";

interface JwtPayload {
  userId: string;
  companyId: string;
  role: string;
  username: string;
}

async function verifyPassword(plainPassword: string, storedPassword: string): Promise<boolean> {
  // Passwords must be bcrypt hashed (starts with $2a$, $2b$, or $2y$)
  if (storedPassword.startsWith("$2")) {
    return bcrypt.compare(plainPassword, storedPassword);
  }
  // Reject any non-hashed passwords for security
  console.warn("Password verification failed: stored password is not a valid bcrypt hash");
  return false;
}

function createToken(user: ExternalUser): string {
  const payload: JwtPayload = {
    userId: user.id,
    companyId: user.company_id,
    role: user.role,
    username: user.username,
  };
  return jwt.sign(payload, JWT_SECRET, { expiresIn: "7d" });
}

function verifyToken(token: string): JwtPayload | null {
  try {
    return jwt.verify(token, JWT_SECRET) as JwtPayload;
  } catch {
    return null;
  }
}

// Login endpoint
router.post("/login", async (req: Request, res: Response) => {
  try {
    const { username, password } = req.body;

    if (!username || !password) {
      return res.status(400).json({ error: "Username and password are required" });
    }

    const user = await findUserByUsername(username);
    if (!user) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const isValidPassword = await verifyPassword(password, user.password);
    if (!isValidPassword) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const company = await findCompanyById(user.company_id);

    const token = createToken(user);

    res.cookie(COOKIE_NAME, token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    });

    return res.json({
      user: {
        id: user.id,
        username: user.username,
        role: user.role,
        companyId: user.company_id,
      },
      company: company ? {
        id: company.id,
        name: company.name,
        logo: company.logo,
        gpuAllocated: company.gpu_allocated,
        cpuAllocated: company.cpu_allocated,
        activeJobs: company.active_jobs,
      } : null,
    });
  } catch (error) {
    console.error("Login error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Get current user
router.get("/me", async (req: Request, res: Response) => {
  try {
    const token = req.cookies[COOKIE_NAME];
    
    if (!token) {
      return res.status(401).json({ error: "Not authenticated" });
    }

    const payload = verifyToken(token);
    if (!payload) {
      return res.status(401).json({ error: "Invalid token" });
    }

    const company = await findCompanyById(payload.companyId);

    return res.json({
      user: {
        id: payload.userId,
        username: payload.username,
        role: payload.role,
        companyId: payload.companyId,
      },
      company: company ? {
        id: company.id,
        name: company.name,
        logo: company.logo,
        gpuAllocated: company.gpu_allocated,
        cpuAllocated: company.cpu_allocated,
        activeJobs: company.active_jobs,
      } : null,
    });
  } catch (error) {
    console.error("Auth me error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Logout endpoint
router.post("/logout", (_req: Request, res: Response) => {
  res.clearCookie(COOKIE_NAME);
  return res.json({ success: true });
});

// Dashboard stats endpoint
router.get("/dashboard-stats", async (req: Request, res: Response) => {
  try {
    const token = req.cookies[COOKIE_NAME];
    
    if (!token) {
      return res.status(401).json({ error: "Not authenticated" });
    }

    const payload = verifyToken(token);
    if (!payload) {
      return res.status(401).json({ error: "Invalid token" });
    }

    const [smilesCount, activeJobs, sshEndpoints, diseasesCount] = await Promise.all([
      getSmilesCountForCompany(payload.companyId),
      getActiveJobsForCompany(payload.companyId),
      getSshConfigsCountForCompany(payload.companyId),
      getDiseasesCount(),
    ]);

    return res.json({
      compoundsInRegistry: smilesCount,
      activeJobs,
      sshEndpoints,
      diseases: diseasesCount,
    });
  } catch (error) {
    console.error("Dashboard stats error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
});

export default router;
