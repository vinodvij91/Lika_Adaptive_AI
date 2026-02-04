#!/bin/bash
# Lika Sciences - DigitalOcean Setup Script

echo "=== Installing Node.js 20 ==="
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
apt-get install -y nodejs

echo "=== Installing dependencies ==="
npm install

echo "=== Building the application ==="
npm run build

echo "=== Setting up PM2 ==="
npm install -g pm2
pm2 start npm --name "lika-sciences" -- start
pm2 save
pm2 startup

echo "=== Setup complete! ==="
echo "Configure nginx for reverse proxy and SSL"
