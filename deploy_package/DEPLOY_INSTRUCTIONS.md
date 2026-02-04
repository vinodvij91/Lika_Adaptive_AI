# Lika Sciences - DigitalOcean Deployment Guide

## Server Requirements
- Ubuntu 22.04 LTS
- Node.js 20.x
- PostgreSQL (DigitalOcean Managed Database)
- Nginx (reverse proxy)
- SSL via Let's Encrypt

## Quick Deployment

### 1. Upload the package to your server
From your PC (where you have SSH keys configured):
```bash
scp -r deploy_package root@167.172.60.179:/root/lika-sciences
```

### 2. SSH into your server
```bash
ssh root@167.172.60.179
cd /root/lika-sciences
```

### 3. Run setup script
```bash
chmod +x setup_do.sh
./setup_do.sh
```

### 4. Configure environment variables
```bash
cp .env.production .env
nano .env  # Fill in your actual values
```

### 5. Configure Nginx
```bash
cp nginx_lika.conf /etc/nginx/sites-available/lika.ai
ln -s /etc/nginx/sites-available/lika.ai /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

### 6. SSL Certificate (Let's Encrypt)
```bash
apt install certbot python3-certbot-nginx
certbot --nginx -d lika.ai -d www.lika.ai
```

### 7. DNS Configuration
Point these records to 167.172.60.179:
- A record: lika.ai -> 167.172.60.179
- A record: www.lika.ai -> 167.172.60.179

## GPU Compute Integration

The platform connects to external GPU nodes for heavy computation:

### Vast.ai (2x RTX 3090)
- SSH: ssh9.vast.ai:37278
- Use your PC's SSH key to connect
- For running drug discovery pipeline with GPU acceleration

### Hetzner (CPU compute)
- IP: 157.180.69.32
- Used for preprocessing and RDKit calculations
- Has Python 3.12 + RDKit installed

## Database Connections

### Primary PostgreSQL (DigitalOcean)
- For user data, projects, molecules, campaigns

### Supabase (Materials)
- 500K+ materials database
- Read-only for materials science queries

### DigitalOcean Spaces
- 1.7M+ SMILES library storage
- S3-compatible API

## Pipeline Execution

Drug discovery and materials science pipelines run on:
1. Hetzner for CPU-bound tasks (RDKit, preprocessing)
2. Vast.ai for GPU-bound tasks (ML, docking)
3. ESMFold API for structure prediction

Run pipelines from your PC:
```bash
ssh root@157.180.69.32
cd /root
python3 lika_drug_discovery_pipeline.py --config disease_discovery_config.yaml
```
