// A styled numeric stepper — replaces the browser's ugly native up/down spinner with
// Factorial-Mono −/+ buttons. Value is clamped to [min, max].
export default function NumberField({ value, onChange, min = 1, max = 999, step = 1, style }) {
  const v = Number(value);
  const clamp = (n) => Math.max(min, Math.min(max, n));
  const set = (n) => onChange(clamp(n));
  return (
    <div className="stepper" style={style}>
      <button type="button" className="stepper-btn" disabled={v <= min}
        onClick={() => set(v - step)} aria-label="decrease">−</button>
      <input className="stepper-input mono" type="text" inputMode="numeric" value={value}
        onChange={(e) => {
          const n = e.target.value.replace(/[^0-9]/g, "");
          onChange(n === "" ? "" : clamp(Number(n)));
        }}
        onBlur={(e) => { if (e.target.value === "") set(min); }} />
      <button type="button" className="stepper-btn" disabled={v >= max}
        onClick={() => set(v + step)} aria-label="increase">+</button>
    </div>
  );
}
