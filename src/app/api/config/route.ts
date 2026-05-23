import { NextResponse } from 'next/server';

/**
 * Config endpoint — tells the frontend whether a server-side OPENCODE_API_KEY is available.
 * The actual key value is never exposed; only a boolean flag is returned.
 */
export async function GET() {
  return NextResponse.json({
    hasServerKey: !!process.env.OPENCODE_API_KEY,
  });
}
