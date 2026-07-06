// A term with a hover/focus explanation card. Wrap a stats label:
//   <InfoTip label="partial η²">explanation… <b>Example:</b> …</InfoTip>
// CSS-only reveal (see index.css .infotip). tabIndex makes it keyboard/touch reachable.
export default function InfoTip({ label, children, align = "left" }) {
  return (
    <span className="infotip" tabIndex={0}>
      {label}<sup className="infotip-i">?</sup>
      <span className={"infotip-card " + align}>{children}</span>
    </span>
  );
}
