// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Integers use bigint, floats use number. Both survive worker structured cloning
// without reserving a captured-object key or rounding integer argument strings.
// YAML temporal scalars use 16-bit units rather than object tags. This preserves their
// type and microseconds through structuredClone/worker messages; captured YAML
// objects cannot accidentally acquire the scalar's representation.
export function yamlTemporal(source, kind = null) {
  const match = source.match(/^(\d{4})-(\d{1,2})-(\d{1,2})(?:[Tt\s]+(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d*))?(?:\s*(Z|([+-])(\d{1,2})(?::(\d{2}))?))?)?$/);
  if (!match) throw new Error('invalid YAML timestamp');
  const [, year, month, day, hour, minute, second, fraction = '', zone, sign, zoneHour, zoneMinute] = match;
  const date = new Date(0);
  date.setUTCFullYear(Number(year), Number(month) - 1, Number(day));
  date.setUTCHours(Number(hour || 0), Number(minute || 0), Number(second || 0), 0);
  if (Number(year) < 1 || date.getUTCFullYear() !== Number(year) || date.getUTCMonth() !== Number(month) - 1 || date.getUTCDate() !== Number(day) || Number(hour || 0) > 23 || Number(minute || 0) > 59 || Number(second || 0) > 59) throw new Error('invalid YAML timestamp');
  const temporalKind = kind || (hour === undefined ? 'date' : 'datetime');
  let text;
  if (temporalKind === 'date') text = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
  else {
    if (Number(zoneHour || 0) * 60 + Number(zoneMinute || 0) >= 24 * 60) throw new Error('invalid YAML timestamp offset');
    const offset = ((Number(zoneHour || 0) * 60 + Number(zoneMinute || 0)) * 60000) * (sign === '-' ? -1 : 1);
    date.setTime(date.getTime() - offset);
    text = date.toISOString().replace(/\.\d{3}Z$/, '') + `.${fraction.slice(0, 6).padEnd(6, '0')}` + (zone ? '+00:00' : '');
  }
  return Uint16Array.from(`${temporalKind}:${text}`, (char) => char.charCodeAt(0));
}

export function temporalText(value) {
  const text = String.fromCharCode(...value);
  if (!/^(?:date|datetime):/.test(text)) throw new Error('unknown typed YAML scalar');
  return text.slice(text.indexOf(':') + 1);
}

// Python's sort_keys orders Unicode scalar values, not UTF-16 code units.
export function compareText(left, right) {
  const a = [...left], b = [...right];
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    const difference = a[index].codePointAt(0) - b[index].codePointAt(0);
    if (difference) return difference < 0 ? -1 : 1;
  }
  return Math.sign(a.length - b.length);
}

export function parseJson(source) {
  if (typeof source !== 'string') throw new TypeError('JSON source must be text');
  let offset = 0;
  const numberToken = /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/y;
  const fail = () => { throw new SyntaxError(`invalid JSON at offset ${offset}`); };
  const space = () => { while (/[\x20\t\r\n]/.test(source[offset] || '\0')) offset++; };
  const string = () => {
    const start = offset++;
    while (offset < source.length) {
      const char = source[offset++];
      if (char === '"') return JSON.parse(source.slice(start, offset));
      if (char.charCodeAt(0) < 32) fail();
      if (char === '\\') {
        const escape = source[offset++];
        if (escape === 'u') {
          if (!/^[0-9a-fA-F]{4}$/.test(source.slice(offset, offset + 4))) fail();
          offset += 4;
        } else if (!escape || !'"\\/bfnrt'.includes(escape)) fail();
      }
    }
    return fail();
  };
  const value = () => {
    space();
    const char = source[offset];
    if (char === '"') return string();
    if (char === '[' || char === '{') {
      const object = char === '{';
      const result = object ? {} : [];
      const end = object ? '}' : ']';
      offset++;
      space();
      if (source[offset] === end) { offset++; return result; }
      for (;;) {
        space();
        if (object) {
          if (source[offset] !== '"') fail();
          const key = string();
          space();
          if (source[offset++] !== ':') fail();
          Object.defineProperty(result, key, { value: value(), writable: true, enumerable: true, configurable: true });
        } else result.push(value());
        space();
        if (source[offset] === end) { offset++; return result; }
        if (source[offset++] !== ',') fail();
      }
    }
    for (const [token, literal] of [['true', true], ['false', false], ['null', null]]) {
      if (source.startsWith(token, offset)) { offset += token.length; return literal; }
    }
    numberToken.lastIndex = offset;
    const token = numberToken.exec(source)?.[0];
    if (!token) return fail();
    offset += token.length;
    return /[.eE]/.test(token) ? Number(token) : BigInt(token);
  };
  const result = value();
  space();
  if (offset !== source.length) fail();
  return result;
}

export function sameValue(left, right) {
  if (typeof left !== typeof right) return false;
  if (left === right) return true;
  if (typeof left === 'number') return Number.isNaN(left) && Number.isNaN(right);
  if (!left || !right || typeof left !== 'object') return false;
  if (ArrayBuffer.isView(left) || ArrayBuffer.isView(right)) return left.constructor === right.constructor && left.length === right.length && left.every((unit, index) => unit === right[index]);
  if (left instanceof Map || right instanceof Map) {
    if (!(left instanceof Map) || !(right instanceof Map) || left.size !== right.size) return false;
    return [...left].every(([key, value]) => [...right].some(([otherKey, otherValue]) => sameValue(key, otherKey) && sameValue(value, otherValue)));
  }
  if (left instanceof Date || right instanceof Date) return left instanceof Date && right instanceof Date && left.getTime() === right.getTime();
  if (Array.isArray(left) !== Array.isArray(right)) return false;
  if (Array.isArray(left)) return left.length === right.length && left.every((item, index) => sameValue(item, right[index]));
  const keys = Object.keys(left);
  return keys.length === Object.keys(right).length && keys.every((key) => Object.hasOwn(right, key) && sameValue(left[key], right[key]));
}

// Display values as data, never as a bigint tag or an internal comparison key.
function floatText(value) {
  if (!Number.isFinite(value)) return String(value);
  if (Object.is(value, -0)) return '-0.0';
  const absolute = Math.abs(value);
  if (absolute && (absolute >= 1e16 || absolute < 1e-4)) {
    return value.toExponential().replace(/e([+-])(\d)$/, 'e$10$2');
  }
  const text = String(value);
  return Number.isInteger(value) ? `${text}.0` : text;
}

export function stringifyValue(value, indent = 2, sortKeys = false) {
  const encode = (item, depth) => {
    if (typeof item === 'bigint') return String(item);
    if (typeof item === 'number') return floatText(item);
    if (item === undefined) return undefined;
    if (!item || typeof item !== 'object') return JSON.stringify(item);
    if (item instanceof Uint16Array) return JSON.stringify(temporalText(item));
    if (item instanceof Uint8Array) return JSON.stringify(btoa(String.fromCharCode(...item)));
    if (item instanceof Date) return JSON.stringify(item.toISOString());
    const array = Array.isArray(item);
    if (item instanceof Map) {
      const entries = [...item].map(([key, value]) => [String(key), value]);
      if (sortKeys) entries.sort(([left], [right]) => compareText(left, right));
      return `{${entries.map(([key, value]) => `${JSON.stringify(key)}:${encode(value, depth + 1)}`).join(',')}}`;
    }
    const keys = array ? [] : Object.keys(item).filter((key) => item[key] !== undefined);
    if (sortKeys) keys.sort(compareText);
    const entries = array ? item.map((child) => encode(child, depth + 1) ?? 'null') : keys.map((key) => `${JSON.stringify(key)}:${indent ? ' ' : ''}${encode(item[key], depth + 1)}`);
    const [open, close] = array ? ['[', ']'] : ['{', '}'];
    if (!entries.length) return open + close;
    if (!indent) return open + entries.join(',') + close;
    const padding = ' '.repeat(indent * (depth + 1));
    return `${open}\n${padding}${entries.join(`,\n${padding}`)}\n${' '.repeat(indent * depth)}${close}`;
  };
  return encode(value, 0);
}

export function canonicalJson(value) { return stringifyValue(value, 0, true); }
