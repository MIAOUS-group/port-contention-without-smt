/**

Functions containing necessary functions for sequential port contention framework.
Basically measure execution time of all pairs of instructions (i1,i2) on two experiments:

 - grouped: {i1,i1,...,i1}{i2,i2,...,i2}
 - interleaved: {i1,i2,i1,i2,...,i1,i2}

If interleaved and grouped have a similar execution time, then i1 and i2 share a cpu port.
If interleaved is faster than grouped, then i1 and i2 use different cpu ports.

This module use wasm functions. You need to build them before using the native wasm_generator.py and makefile.

**/


/**
 * warmUp - Executes arbitrary computations to speed up the frequency of the CPU and disminish the impact of the scaling governor.
 * This lets our experiment work better when the frequency is unpinned.
 *
 */
async function warmUp() {
  let {grouped, interleaved} = await initWasm("i64.ctz","i64.clz");
  for (var i = 0; i < 10;i++) {
  grouped(BigInt(1256456456));
  interleaved(BigInt(23415646514));}
}

/**
 * initWasm - Instantiates the experiment functions {grouped,interleaved} for the two instructions given in parameter.
 * This function is asynchronous, use it with async/await.
 *
 * @param  {string} instruction_1 First tested instruction
 * @param  {string} instruction_2 Second tested instruction

 * @return {Object} The two experimental WebAssembly functions grouped and interleaved for isntruction_1 and instruction_2 ready to be executed.
 */
async function initWasm(instruction_1, instruction_2) {
  const wasm = fetch(`./build/${instruction_1}_${instruction_2}.wasm`);
  const {instance} = await WebAssembly.instantiateStreaming(wasm);
  let grouped = await instance.exports.grouped;
  let interleaved = await instance.exports.interleaved;
  return {grouped, interleaved}
}

/**
 * getParam - Creates a random parameter for the WebAssembly functions.
 * As our functions are automatically generated with various input types, we need to create a proper parameter
 *
 * @param  {string} instruction_1 First tested instruction
 * @return           The parameter, or array of parameters for vectorial cases
 */
function getParam(instruction_1) {
  if (VOP.includes(instruction_1)) {
    var param = []
    var {numType, paramCount} = parseVShape(instruction_1);
    switch (numType) {
      case 'i16':
        for (var i = 0; i < Number(paramCount); i++) {
          param.push(getRandomInt(Math.pow(2,15)))
        }
        break;
      case 'f64':
        for (var i = 0; i < Number(paramCount); i++) {
          param.push(getRandomFloat(Math.pow(2,63)));
        }
        break;
      default:
        console.log("Invalid type");
    }
  }
  else{
    if (typeof(instruction_1) == "string") {
      var type = instruction_1.substring(0,3)
    }
    else {
      var type = instruction_1[1].substring(0,3)
    }
    var param
    switch (type) { // As we have different types, we instantiate the proper parameter
      case 'i32':
        param = getRandomInt(Math.pow(2,31));
        break;
      case 'i64':
        param = BigInt(getRandomInt(Math.pow(2,63)));
        break;

      case 'f32':
        param = getRandomFloat(Math.pow(2,30));
        break;

      case 'f64':
        param = getRandomFloat(Math.pow(2,30));
        break;
      default:
        console.log("Invalid type");
    }
  }
  return param
}


/**
 * getParam - Creates a random parameter for the WebAssembly functions.
 * As our functions are automatically generated with various input types, we need to create a proper parameter
 *
 * @param  {string} instruction_1 First tested instruction
 * @return           The parameter, or array of parameters for vectorial cases
 */
async function testSeqPCWithClock(clock, instruction_1, instruction_2) {
  console.log(`Testing ${instruction_1}_${instruction_2}.wasm`)
  let {grouped, interleaved} = await initWasm(instruction_1, instruction_2);
  var param = getParam(instruction_1);
  let begin,end;
  groupedTime = []
  interleavedTime = []

  for (var i = 0; i <100; i++) {
    begin = Atomics.load(clock.array,0);
    interleaved(param)
    end = Atomics.load(clock.array,0);
    interleavedTime.push(end-begin);

    begin = Atomics.load(clock.array,0);
    grouped(param)
    end = Atomics.load(clock.array,0);
    groupedTime.push(end-begin);
  }
  var grouped_median = math.median(groupedTime)
  var interleaved_median = math.median(interleavedTime)
  console.log("Grouped: ", grouped_median)
  console.log("Interleaved: ", interleaved_median)
  console.log("Ratio: ", grouped_median / interleaved_median);
  return {grouped_median, interleaved_median}
}


/**
 * testAll - Test all pairs of instructions defined in instructions.js
 * This is our communication with the selenium framework, and the json returned contains all results.
 *
 * @return {Object} A JSON object with timings for all pairs.
 */
async function testAll() {
  let clock = await initSAB(atomic = true);
  console.log("Warming up");
  await warmUp();
  results = {}

  type = ["i64"]
  instruction_list = cross_product(type, IUNOP).concat(cross_product(type,IBINOP))
  for (instruction_1 of instruction_list){
      for (instruction_2 of instruction_list){
        if (!((`${instruction_1}_${instruction_2}` in results) || (`${instruction_2}_${instruction_1}` in results))) {
          var {grouped_median, interleaved_median} = await testSeqPCWithClock(clock,instruction_1,instruction_2)
          results[`${instruction_1}_${instruction_2}`] = {"grouped": grouped_median, "interleaved": interleaved_median}
        }
      }
  }

  type = ["f64"]
  instruction_list = cross_product(type, FUNOP).concat(cross_product(type,FBINOP))
  for (instruction_1 of instruction_list)
      for (instruction_2 of instruction_list)
          if (!((`${instruction_1}_${instruction_2}` in results) || (`${instruction_2}_${instruction_1}` in results))) {
            var {grouped_median, interleaved_median} = await testSeqPCWithClock(clock,instruction_1,instruction_2)
            results[`${instruction_1}_${instruction_2}`] = {"grouped": grouped_median, "interleaved": interleaved_median}
          }

  type = ["i16x8"]
  instruction_list = cross_product(type, VIUNOP).concat(cross_product(type,VIBINOP),cross_product(type,VIMINMAXOP),cross_product(type,VISATBINOP),cross_product(type,["mul"]),cross_product(type,["avgr_u"]), ["i16x8.q15mulr_sat_s"])
  for (instruction_1 of instruction_list) {
      for (instruction_2 of instruction_list) {
          if (!((`${instruction_1}_${instruction_2}` in results) || (`${instruction_2}_${instruction_1}` in results))) {
            var {grouped_median, interleaved_median} = await testSeqPCWithClock(clock,instruction_1,instruction_2)
            results[`${instruction_1}_${instruction_2}`] = {"grouped": grouped_median, "interleaved": interleaved_median}
        }
      }
  }

  type = ["f64x2"]
  instruction_list = cross_product(type, VFUNOP).concat(cross_product(type,VFBINOP))
  for (instruction_1 of instruction_list) {
      for (instruction_2 of instruction_list) {
        if (!((`${instruction_1}_${instruction_2}` in results) || (`${instruction_2}_${instruction_1}` in results))) {

          var {grouped_median, interleaved_median} = await testSeqPCWithClock(clock,instruction_1,instruction_2)
          results[`${instruction_1}_${instruction_2}`] = {"grouped": grouped_median, "interleaved": interleaved_median}
        }
      }
  }


  console.log("done")
  console.log(results)
  return results
}
