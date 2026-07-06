// The CAFE mark: a coffee mug with factorial-branch nodes (amber) + a check. Echoes the full logo.
export default function MugIcon({ size = 34 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* branch stems */}
      <path d="M20 25c0-5 4-6 4-11M24 25c0-4 3-5 6-7M20 25c0-4-2-5-5-6"
            stroke="#e5e2e1" strokeWidth="2" strokeLinecap="round" fill="none" />
      {/* mug body */}
      <path d="M13 24h18v6a9 9 0 0 1-9 9h0a9 9 0 0 1-9-9v-6Z"
            stroke="#e5e2e1" strokeWidth="2.2" strokeLinejoin="round" fill="none" />
      {/* handle */}
      <path d="M31 26h3a4 4 0 0 1 0 8h-3" stroke="#e5e2e1" strokeWidth="2.2" strokeLinecap="round" fill="none" />
      {/* nodes */}
      <circle cx="24" cy="12" r="3.4" fill="#ffbf00" />
      <circle cx="31" cy="17" r="3" fill="#ffbf00" />
      <circle cx="14.5" cy="18" r="3" fill="#ffbf00" />
      {/* check badge */}
      <circle cx="22" cy="31" r="5.5" fill="#131313" stroke="#e5e2e1" strokeWidth="1.6" />
      <path d="M19.6 31.2l1.7 1.7 3-3.4" stroke="#ffbf00" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}
