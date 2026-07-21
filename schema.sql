-- Run this in Supabase SQL Editor before deploying

create table if not exists proofread_jobs (
  id uuid primary key default gen_random_uuid(),
  original_text text not null,
  corrected_text text not null,
  changes jsonb default '[]'::jsonb,
  summary text,
  stats jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

-- Migration for tables created before the stats column existed:
alter table proofread_jobs add column if not exists stats jsonb default '{}'::jsonb;

alter table proofread_jobs enable row level security;

-- service_role key (used by backend) bypasses RLS by default.
-- This policy only matters if you later expose anon/public reads.
create policy "service role full access"
  on proofread_jobs
  for all
  using (true)
  with check (true);
