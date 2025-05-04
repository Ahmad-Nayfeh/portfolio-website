/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: false, // Keep as false to enforce linting
  },
  typescript: {
    ignoreBuildErrors: false, // Keep as false to enforce type checking
  },
  // `serverExternalPackages` is the correct top-level key now
  serverExternalPackages: [],
  experimental: {
    // Options still under experimental go here
    typedRoutes: true,
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  pageExtensions: ['ts', 'tsx', 'js', 'jsx'],
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production' ? { exclude: ['error', 'warn'] } : false,
  }
}

export default nextConfig