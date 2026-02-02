import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, HeadObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

export interface SpacesConfig {
  endpoint: string;
  region: string;
  bucket: string;
  accessKeyId: string;
  secretAccessKey: string;
  cdnEndpoint?: string;
}

export interface UploadResult {
  key: string;
  bucket: string;
  size: number;
  etag?: string;
  cdnUrl?: string;
  storageUrl: string;
}

export interface SignedUrlOptions {
  expiresIn?: number;
  contentType?: string;
}

function getSpacesConfig(): SpacesConfig | null {
  const endpoint = process.env.DO_SPACES_ENDPOINT;
  const region = process.env.DO_SPACES_REGION || "nyc3";
  const bucket = process.env.DO_SPACES_BUCKET;
  const accessKeyId = process.env.DO_SPACES_ACCESS_KEY_ID;
  const secretAccessKey = process.env.DO_SPACES_SECRET_ACCESS_KEY;
  const cdnEndpoint = process.env.DO_SPACES_CDN_ENDPOINT;

  if (!endpoint || !bucket || !accessKeyId || !secretAccessKey) {
    return null;
  }

  return { endpoint, region, bucket, accessKeyId, secretAccessKey, cdnEndpoint };
}

function createS3Client(config: SpacesConfig): S3Client {
  return new S3Client({
    endpoint: config.endpoint,
    region: config.region,
    credentials: {
      accessKeyId: config.accessKeyId,
      secretAccessKey: config.secretAccessKey,
    },
    forcePathStyle: false,
  });
}

export function generateAssetKey(
  moleculeId: string,
  assetType: string,
  extension: string,
  companyId?: string
): string {
  const prefix = companyId ? `companies/${companyId}` : "global";
  const hash = crypto.createHash("md5").update(moleculeId).digest("hex").slice(0, 8);
  return `${prefix}/molecules/${hash}/${moleculeId}/${assetType}.${extension}`;
}

export async function uploadAssetFromFile(
  filePath: string,
  key: string,
  mimeType: string,
  deleteAfterUpload: boolean = true
): Promise<UploadResult> {
  const config = getSpacesConfig();
  if (!config) {
    throw new Error("DO Spaces not configured. Set DO_SPACES_* environment variables.");
  }

  const client = createS3Client(config);
  const fileBuffer = fs.readFileSync(filePath);
  const size = fileBuffer.length;

  const command = new PutObjectCommand({
    Bucket: config.bucket,
    Key: key,
    Body: fileBuffer,
    ContentType: mimeType,
    ACL: "public-read",
  });

  const response = await client.send(command);

  if (deleteAfterUpload && fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }

  const storageUrl = `${config.endpoint}/${config.bucket}/${key}`;
  const cdnUrl = config.cdnEndpoint
    ? `${config.cdnEndpoint}/${key}`
    : storageUrl;

  return {
    key,
    bucket: config.bucket,
    size,
    etag: response.ETag,
    cdnUrl,
    storageUrl,
  };
}

export async function uploadAssetFromBuffer(
  buffer: Buffer,
  key: string,
  mimeType: string
): Promise<UploadResult> {
  const config = getSpacesConfig();
  if (!config) {
    throw new Error("DO Spaces not configured. Set DO_SPACES_* environment variables.");
  }

  const client = createS3Client(config);

  const command = new PutObjectCommand({
    Bucket: config.bucket,
    Key: key,
    Body: buffer,
    ContentType: mimeType,
    ACL: "public-read",
  });

  const response = await client.send(command);

  const storageUrl = `${config.endpoint}/${config.bucket}/${key}`;
  const cdnUrl = config.cdnEndpoint
    ? `${config.cdnEndpoint}/${key}`
    : storageUrl;

  return {
    key,
    bucket: config.bucket,
    size: buffer.length,
    etag: response.ETag,
    cdnUrl,
    storageUrl,
  };
}

export async function getSignedDownloadUrl(
  key: string,
  options: SignedUrlOptions = {}
): Promise<string> {
  const config = getSpacesConfig();
  if (!config) {
    throw new Error("DO Spaces not configured. Set DO_SPACES_* environment variables.");
  }

  const client = createS3Client(config);
  const expiresIn = options.expiresIn || 3600;

  const command = new GetObjectCommand({
    Bucket: config.bucket,
    Key: key,
  });

  return getSignedUrl(client, command, { expiresIn });
}

export async function getSignedUploadUrl(
  key: string,
  mimeType: string,
  options: SignedUrlOptions = {}
): Promise<string> {
  const config = getSpacesConfig();
  if (!config) {
    throw new Error("DO Spaces not configured. Set DO_SPACES_* environment variables.");
  }

  const client = createS3Client(config);
  const expiresIn = options.expiresIn || 3600;

  const command = new PutObjectCommand({
    Bucket: config.bucket,
    Key: key,
    ContentType: mimeType,
  });

  return getSignedUrl(client, command, { expiresIn });
}

export async function deleteAsset(key: string): Promise<void> {
  const config = getSpacesConfig();
  if (!config) {
    throw new Error("DO Spaces not configured. Set DO_SPACES_* environment variables.");
  }

  const client = createS3Client(config);

  const command = new DeleteObjectCommand({
    Bucket: config.bucket,
    Key: key,
  });

  await client.send(command);
}

export async function assetExists(key: string): Promise<boolean> {
  const config = getSpacesConfig();
  if (!config) {
    return false;
  }

  const client = createS3Client(config);

  try {
    const command = new HeadObjectCommand({
      Bucket: config.bucket,
      Key: key,
    });
    await client.send(command);
    return true;
  } catch (error: any) {
    if (error.name === "NotFound" || error.$metadata?.httpStatusCode === 404) {
      return false;
    }
    throw error;
  }
}

export async function getAssetAsBuffer(key: string): Promise<Buffer> {
  const config = getSpacesConfig();
  if (!config) {
    throw new Error("DO Spaces not configured. Set DO_SPACES_* environment variables.");
  }

  const client = createS3Client(config);

  const command = new GetObjectCommand({
    Bucket: config.bucket,
    Key: key,
  });

  const response = await client.send(command);
  
  if (!response.Body) {
    throw new Error(`No content found for key: ${key}`);
  }

  const chunks: Uint8Array[] = [];
  for await (const chunk of response.Body as AsyncIterable<Uint8Array>) {
    chunks.push(chunk);
  }
  
  return Buffer.concat(chunks);
}

export async function getAssetAsString(key: string): Promise<string> {
  const buffer = await getAssetAsBuffer(key);
  return buffer.toString("utf-8");
}

export function getCdnUrl(key: string): string | null {
  const config = getSpacesConfig();
  if (!config) return null;

  if (config.cdnEndpoint) {
    return `${config.cdnEndpoint}/${key}`;
  }
  return `${config.endpoint}/${config.bucket}/${key}`;
}

export function isSpacesConfigured(): boolean {
  return getSpacesConfig() !== null;
}

export function getMimeTypeForAsset(assetType: string): string {
  const mimeTypes: Record<string, string> = {
    thumbnail_2d: "image/png",
    thumbnail_3d: "image/png",
    conformer_sdf: "chemical/x-mdl-sdfile",
    conformer_pdb: "chemical/x-pdb",
    descriptors_json: "application/json",
    fingerprint_json: "application/json",
  };
  return mimeTypes[assetType] || "application/octet-stream";
}

export function getExtensionForAsset(assetType: string): string {
  const extensions: Record<string, string> = {
    thumbnail_2d: "png",
    thumbnail_3d: "png",
    conformer_sdf: "sdf",
    conformer_pdb: "pdb",
    descriptors_json: "json",
    fingerprint_json: "json",
  };
  return extensions[assetType] || "bin";
}
