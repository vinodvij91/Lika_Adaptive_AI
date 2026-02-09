import express, { type Request, Response, NextFunction } from "express";
import cookieParser from "cookie-parser";
import { registerRoutes } from "./routes";
import { serveStatic } from "./static";
import { createServer } from "http";
import authRoutes from "./auth-routes";
import { storage } from "./storage";
import { seedDemoData } from "./demo-data-seeder";
import { auth, requiresAuth } from "express-openid-connect";
import { syncAuth0UserAndResolveTenant, lookupTenantForUser } from "./tenant-sync";

const app = express();
const httpServer = createServer(app);

declare module "http" {
  interface IncomingMessage {
    rawBody: unknown;
  }
}

app.use(cookieParser());
app.use(
  express.json({
    verify: (req, _res, buf) => {
      req.rawBody = buf;
    },
  }),
);

app.use(express.urlencoded({ extended: false }));

const isProduction = process.env.NODE_ENV === "production";
const replitDevDomain = process.env.REPLIT_DEV_DOMAIN;
const replitDeploymentUrl = process.env.REPLIT_DEPLOYMENT_URL;

let baseURL: string;
if (isProduction && replitDeploymentUrl) {
  baseURL = `https://${replitDeploymentUrl}`;
} else if (replitDevDomain) {
  baseURL = `https://${replitDevDomain}`;
} else {
  baseURL = "http://localhost:5000";
}

const sessionSecret = process.env.SESSION_SECRET;
if (!sessionSecret && isProduction) {
  throw new Error("SESSION_SECRET environment variable is required in production");
}

const auth0Enabled = !!(process.env.AUTH0_DOMAIN && process.env.AUTH0_CLIENT_ID && process.env.AUTH0_CLIENT_SECRET);

if (auth0Enabled) {
  const auth0Config = {
    authRequired: false,
    auth0Logout: true,
    secret: sessionSecret || require("crypto").randomBytes(32).toString("hex"),
    baseURL,
    clientID: process.env.AUTH0_CLIENT_ID!,
    clientSecret: process.env.AUTH0_CLIENT_SECRET!,
    issuerBaseURL: `https://${process.env.AUTH0_DOMAIN}`,
    routes: {
      login: "/api/auth/login",
      logout: "/api/auth/logout",
      callback: "/api/auth/callback",
      postLogoutRedirect: "/",
    },
    afterCallback: async (_req: Request, _res: Response, session: any) => {
      try {
        let sub = "";
        let email = "";
        let name = "";

        if (session.id_token) {
          const payload = session.id_token.split(".")[1];
          const decoded = Buffer.from(payload, "base64url").toString("utf-8");
          const claims = JSON.parse(decoded);
          sub = claims.sub || "";
          email = claims.email || "";
          name = claims.name || claims.nickname || "";
        }

        if (sub && email) {
          const tenant = await syncAuth0UserAndResolveTenant(sub, email, name);
          session.tenantId = tenant.tenantId;
          session.tenantRole = tenant.role;
        }
      } catch (err) {
        console.error("afterCallback tenant sync error:", err);
      }
      return session;
    },
  };

  app.use(auth(auth0Config));
}

app.get("/api/auth/me", async (req: Request, res: Response) => {
  if ((req as any).oidc?.isAuthenticated()) {
    const user = (req as any).oidc.user;
    const userId = user.sub;

    let tenantId = (req as any).appSession?.tenantId;
    let role = (req as any).appSession?.tenantRole;

    if (!tenantId) {
      const tenant = await lookupTenantForUser(userId);
      tenantId = tenant.tenantId;
      role = tenant.role;
    }

    return res.json({
      authenticated: true,
      id: userId,
      userId: userId,
      email: user.email,
      firstName: user.given_name || user.nickname || user.name?.split(" ")[0] || "",
      lastName: user.family_name || user.name?.split(" ").slice(1).join(" ") || "",
      profileImageUrl: user.picture || null,
      tenantId: tenantId || "jnj",
      role: role || "scientist",
      createdAt: new Date(),
      updatedAt: new Date(),
    });
  }
  return res.status(401).json({ authenticated: false });
});

const PUBLIC_API_PATHS = [
  "/api/auth/",
  "/api/ext-auth/",
];

app.use("/api", async (req: Request, res: Response, next: NextFunction) => {
  const fullPath = req.baseUrl + req.path;
  if (PUBLIC_API_PATHS.some(p => fullPath.startsWith(p))) {
    return next();
  }

  if (!auth0Enabled) {
    return next();
  }

  if (!(req as any).oidc?.isAuthenticated()) {
    return res.status(401).json({ error: "Not authenticated" });
  }

  const userId = (req as any).oidc.user?.sub;
  const email = (req as any).oidc.user?.email;

  let tenantId = (req as any).appSession?.tenantId;
  let role = (req as any).appSession?.tenantRole;

  if (!tenantId && userId) {
    const tenant = await lookupTenantForUser(userId);
    tenantId = tenant.tenantId;
    role = tenant.role;
  }

  (req as any).user = {
    id: userId,
    email: email,
    tenantId: tenantId || "jnj",
    role: role || "scientist",
  };

  next();
});

// External auth routes (for DigitalOcean database - legacy fallback)
app.use("/api/ext-auth", authRoutes);

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  await registerRoutes(httpServer, app);

  try {
    await storage.seedBuiltInTemplates();
    log("Built-in pipeline templates seeded");
  } catch (err) {
    log("Failed to seed pipeline templates (may already exist)");
  }

  try {
    await seedDemoData();
    log("Demo data seeded successfully");
  } catch (err) {
    log("Failed to seed demo data: " + (err instanceof Error ? err.message : "Unknown error"));
  }

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (process.env.NODE_ENV === "production") {
    serveStatic(app);
  } else {
    const { setupVite } = await import("./vite");
    await setupVite(httpServer, app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || "5000", 10);
  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    () => {
      log(`serving on port ${port}`);
    },
  );
})();
